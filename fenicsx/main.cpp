/* Include relevant header files
"poisson.h" was created by
ffcx poisson.py
*/
#include "poisson.h"
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/petsc.h>
#include <utility>
#include <vector>

using namespace dolfinx;
using T = PetscScalar;
using U = typename dolfinx::scalar_value_type_t<T>;

int main(int argc, char *argv[]) {
  dolfinx::init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);

  {
    // Create unit cube mesh and function space (specified in poisson.py)
    auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
    auto mesh = std::make_shared<mesh::Mesh<U>>(
        mesh::create_box<U>(MPI_COMM_WORLD, {{{0, 0, 0}, {1, 1, 1}}},
                            {15, 15, 15}, mesh::CellType::tetrahedron, part));

    auto V = std::make_shared<fem::FunctionSpace<U>>(
        fem::create_functionspace(functionspace_form_poisson_a, "u", mesh));

    // Define coefficients and weak form
    auto kappa = std::make_shared<fem::Constant<T>>(1.0);
    auto f = std::make_shared<fem::Function<T>>(V);
    auto g = std::make_shared<fem::Function<T>>(V);

    auto a = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_poisson_a, {V, V}, {}, {{"kappa", kappa}}, {}));

    auto L = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_poisson_L, {V}, {{"f", f}, {"g", g}}, {}, {}));

    // Specify boundary conditions
    auto facets = mesh::locate_entities_boundary(*mesh, 2, [](auto x) {
      using U = typename decltype(x)::value_type;
      constexpr U eps = 1.0e-8;
      std::vector<std::int8_t> marker(x.extent(1), false);
      for (std::size_t p = 0; p < x.extent(1); ++p) {
        auto x0 = x(0, p);
        auto x1 = x(1, p);
        auto x2 = x(2, p);
        auto tol = 1 - eps;
        if (std::abs(x0) > tol or std::abs(x1) > tol or std::abs(x2) > tol)
          marker[p] = true;
      }
      return marker;
    });
    const auto bdofs = fem::locate_dofs_topological(
        *V->mesh()->topology_mutable(), *V->dofmap(), 2, facets);
    auto bc = std::make_shared<const fem::DirichletBC<T>>(0.0, bdofs, V);

    f->interpolate(
        [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>> {
          std::vector<T> f;
          for (std::size_t p = 0; p < x.extent(1); ++p) {
            auto dx = (x(0, p) - 0.40) * (x(0, p) - 0.40);
            auto dy = (x(1, p) - 0.55) * (x(1, p) - 0.55);
            auto dz = (x(2, p) - 0.27) * (x(2, p) - 0.27);
            f.push_back(10 * std::exp(2. * (dx + dy + dz)));
          }

          return {f, {f.size()}};
        });

    g->interpolate(
        [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>> {
          std::vector<T> f;
          for (std::size_t p = 0; p < x.extent(1); ++p)
            f.push_back(0);
          return {f, {f.size()}};
        });

    // Bilinear and Linear side assembly
    auto u = std::make_shared<fem::Function<T>>(V);
    auto A = la::petsc::Matrix(fem::petsc::create_matrix(*a), false);
    la::Vector<T> b(L->function_spaces()[0]->dofmap()->index_map,
                    L->function_spaces()[0]->dofmap()->index_map_bs());

    MatZeroEntries(A.mat());
    fem::assemble_matrix(la::petsc::Matrix::set_block_fn(A.mat(), ADD_VALUES),
                         *a, {bc});
    MatAssemblyBegin(A.mat(), MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(A.mat(), MAT_FLUSH_ASSEMBLY);
    fem::set_diagonal<T>(la::petsc::Matrix::set_fn(A.mat(), INSERT_VALUES), *V,
                         {bc});
    MatAssemblyBegin(A.mat(), MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A.mat(), MAT_FINAL_ASSEMBLY);

    b.set(0.0);
    fem::assemble_vector(b.mutable_array(), *L);
    fem::apply_lifting<T, U>(b.mutable_array(), {a}, {{bc}}, {}, T(1));
    b.scatter_rev(std::plus<T>());
    fem::set_bc<T, U>(b.mutable_array(), {bc});

    // Solver (with preconditioner) setup
    la::petsc::KrylovSolver lu(MPI_COMM_WORLD);
    la::petsc::options::set("ksp_type", "cg");
    la::petsc::options::set("pc_type", "gamg");
    lu.set_from_options();

    // Solve the problem
    lu.set_operator(A.mat());
    la::petsc::Vector _u(la::petsc::create_vector_wrap(*u->x()), false);
    la::petsc::Vector _b(la::petsc::create_vector_wrap(b), false);
    lu.solve(_u.vec(), _b.vec());

    u->x()->scatter_fwd();

    // Save solution in VTK format
    io::VTKFile file(MPI_COMM_WORLD, "u.pvd", "w");
    file.write<T>({*u}, 0.0);
  }

  PetscFinalize();

  return 0;
}
