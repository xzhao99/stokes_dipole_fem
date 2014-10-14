/* The libMesh Finite Element Library. */
/* Copyright (C) 2003  Benjamin S. Kirk */

/* This library is free software; you can redistribute it and/or */
/* modify it under the terms of the GNU Lesser General Public */
/* License as published by the Free Software Foundation; either */
/* version 2.1 of the License, or (at your option) any later version. */

/* This library is distributed in the hope that it will be useful, */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU */
/* Lesser General Public License for more details. */

/* You should have received a copy of the GNU Lesser General Public */
/* License along with this library; if not, write to the Free Software */
/* Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA */

// <h1>Systems Example 1 - Stokes Equations with pair force </h1>
//
// This example shows how a simple, linear system of equations
// can be solved in parallel.  The system of equations are the familiar
// Stokes equations for low-speed incompressible fluid flow.

// C++ include files that we need
#include <iostream>
#include <algorithm>
#include <math.h>

// Basic include file needed for the mesh functionality.
#include "libmesh/libmesh.h"
#include "libmesh/mesh.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/exodusII_io.h"
#include "libmesh/equation_systems.h"
#include "libmesh/fe.h"
#include "libmesh/quadrature_gauss.h"
#include "libmesh/dof_map.h"
#include "libmesh/elem.h"
#include "libmesh/sparse_matrix.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/dense_matrix.h"
#include "libmesh/dense_vector.h"
#include "libmesh/linear_implicit_system.h"

// For systems of equations the DenseSubMatrix and DenseSubVector provide convenient ways
// for assembling the element matrix and vector on a component-by-component basis.
#include "libmesh/dense_submatrix.h"
#include "libmesh/dense_subvector.h"



// Bring in everything from the libMesh namespace
using namespace libMesh;


// include user defined classes
#include "polymer_chain.h"    // //class PolymerChain;



// Function prototype.  This function will assemble the system matrix and RHS.
void assemble_stokes (EquationSystems& es,
                      const std::string& system_name);


/// ======================================================================================= ///
/// ======================================================================================= ///
/// ======================================================================================= ///
// The main program.
int main (int argc, char** argv)
{
  // Initialize libMesh.
  LibMeshInit init (argc, argv);

  // Skip this 2D example if libMesh was compiled as 1D-only.
  libmesh_example_requires(2 <= LIBMESH_DIM, "2D support");

  // This example NaNs with the Eigen sparse linear solvers and Trilinos solvers,
  // but should work OK with either PETSc or Laspack.
  libmesh_example_requires(libMesh::default_solver_package() != EIGEN_SOLVERS, "--enable-petsc or --enable-laspack");
  libmesh_example_requires(libMesh::default_solver_package() != TRILINOS_SOLVERS, "--enable-petsc or --enable-laspack");

  // Create a mesh, with dimension to be overridden later, distributed across the default MPI communicator.
  Mesh mesh(init.comm());

  // Use the MeshTools::Generation mesh generator to create a uniform 2D grid on the square [-1,1]^2.
  // We instruct the mesh generator to build a mesh of 8x8 Quad9 elements.
  MeshTools::Generation::build_square (mesh, 100, 50, -1., 1., -0.5, 0.5, QUAD9);

  // Print information about the mesh to the screen.
  mesh.print_info();

  // Create an equation systems object.
  EquationSystems equation_systems (mesh);

  // Declare the system and its variables. Create a system named "Stokes"
  LinearImplicitSystem & system =
    equation_systems.add_system<LinearImplicitSystem> ("Stokes");

  // Add the variables "u" & "v" to "Stokes" using second-order approximation.
  // Add the variable "p" to "Stokes" with a first-order basis, satisfying the LBB condition
  system.add_variable ("u", SECOND);
  system.add_variable ("v", SECOND);
  system.add_variable ("p", FIRST);

  // Give the system a pointer to the matrix assembly function.
  system.attach_assemble_function (assemble_stokes);

  // Initialize the data structures for the equation system.
  equation_systems.init ();

  equation_systems.parameters.set<unsigned int>("linear solver maximum iterations") = 250;
  equation_systems.parameters.set<Real>        ("linear solver tolerance") = TOLERANCE;

  // Prints information about the system to the screen.
  // equation_systems.print_info();

  // Assemble & solve the linear system, then write the solution.
  equation_systems.get_system("Stokes").solve();

#ifdef LIBMESH_HAVE_EXODUS_API
  ExodusII_IO(mesh).write_equation_systems ("out.e", equation_systems);
#endif // #ifdef LIBMESH_HAVE_EXODUS_API

  // All done.
  return 0;
}


/// ======================================================================================= ///
/// ======================================================================================= ///
/// ======================================================================================= ///
void assemble_stokes (EquationSystems& es,
                      const std::string& system_name)
{
  // It is a good idea to make sure we are assembling the proper system.
  libmesh_assert_equal_to (system_name, "Stokes");

  // Get a constant reference to the mesh object.
  const MeshBase& mesh = es.get_mesh();

  // The dimension that we are running
  const unsigned int dim = mesh.mesh_dimension();
  
  // define polymer chain
  PolymerChain polymer_chain_two_beads;
  polymer_chain_two_beads.init(mesh);
  polymer_chain_two_beads.print_info();

  // Get a reference to the Convection-Diffusion system object.
  LinearImplicitSystem & system =
        es.get_system<LinearImplicitSystem> ("Stokes");

  // Numeric ids corresponding to each variable in the system
  const unsigned int u_var = system.variable_number ("u");
  const unsigned int v_var = system.variable_number ("v");
  const unsigned int p_var = system.variable_number ("p");
    
  /// ----------------------------------- My output -----------------------------------
  std::cout<< "u_var = " << u_var << ", v_var = " << v_var << ", p_var = " << p_var << std::endl;
  /// ----------------------------------- My output -----------------------------------

  // Get the Finite Element type for "u" and "p".  Note "u" is the same as the type for "v".
  FEType fe_vel_type = system.variable_type(u_var);
  FEType fe_pres_type = system.variable_type(p_var);

  // Build a Finite Element object of the specified type for the velocity and pressure variables.
  AutoPtr<FEBase> fe_vel  (FEBase::build(dim, fe_vel_type));
  AutoPtr<FEBase> fe_pres (FEBase::build(dim, fe_pres_type));

  // A Gauss quadrature rule for numerical integration.
  // Let the FEType object decide what order rule is appropriate.
  QGauss qrule (dim, fe_vel_type.default_quadrature_order());

  // Tell the finite element objects to use our quadrature rule.
  fe_vel->attach_quadrature_rule (&qrule);
  fe_pres->attach_quadrature_rule (&qrule);

  // The element Jacobian * quadrature weight at each integration point.
  const std::vector<Real>& JxW = fe_vel->get_JxW();
    
  // The element shape function gradients for the velocity
  // variables evaluated at the quadrature points.
  const std::vector<std::vector<RealGradient> >& dphi = fe_vel->get_dphi();
  
  // The element shape functions for the pressure variable evaluated at the quad pts.
  const std::vector<std::vector<Real> >& psi = fe_pres->get_phi();
  
  /// ----------------------------------- My output -----------------------------------
  // NOTE: here JxW and dphi and other element quantities are not computed
  // until fe_vel->reinit  (elem); and fe_pres->reinit (elem); for a specified element.
//  std::cout << "the size of JxW vector is :"<< JxW.size() << std::endl;
//  std::cout << "the size of dphi matrix is :"<<dphi.size()<< std::endl;
//  std::cout << "the size of psi matrix is :"<< psi.size() << std::endl;
  /// ----------------------------------- My output -----------------------------------

  // A reference to the DofMap object for this system.  The DofMap object handles
  // the index translation from node and element numbers to degree of freedom numbers.
  const DofMap & dof_map = system.get_dof_map();

  // Define data structures to contain the element matrix and RHS vector contribution Ke and Fe.
  DenseMatrix<Number> Ke;
  DenseVector<Number> Fe;

  DenseSubMatrix<Number>
    Kuu(Ke), Kuv(Ke), Kup(Ke),
    Kvu(Ke), Kvv(Ke), Kvp(Ke),
    Kpu(Ke), Kpv(Ke), Kpp(Ke);

  DenseSubVector<Number>  Fu(Fe),    Fv(Fe),    Fp(Fe);

  // This vector will hold the degree of freedom indices for the element.
  // These define where in the global system the element degrees of freedom get mapped.
  std::vector<dof_id_type> dof_indices;
  std::vector<dof_id_type> dof_indices_u;
  std::vector<dof_id_type> dof_indices_v;
  std::vector<dof_id_type> dof_indices_p;

  // Now we will loop over all the elements in the mesh that live on the local processor.
  // We will compute the element matrix and RHS contribution.
  // In case users later modify this program to include refinement, we will be safe and
  // will only consider the active elements; hence we use a variant of the active_elem_iterator.
  MeshBase::const_element_iterator       el     = mesh.active_local_elements_begin();
  const MeshBase::const_element_iterator end_el = mesh.active_local_elements_end();
    
  dof_id_type num_elem = 0;
  for ( ; el != end_el; ++el)
    {
      // Store a pointer to the element we are currently working on.
      const Elem* elem = *el;

      // Get the degree of freedom indices for the current element.
      dof_map.dof_indices (elem, dof_indices);
      dof_map.dof_indices (elem, dof_indices_u, u_var);
      dof_map.dof_indices (elem, dof_indices_v, v_var);
      dof_map.dof_indices (elem, dof_indices_p, p_var);

      const unsigned int n_dofs   = dof_indices.size();
      const unsigned int n_u_dofs = dof_indices_u.size();
      const unsigned int n_v_dofs = dof_indices_v.size();
      const unsigned int n_p_dofs = dof_indices_p.size();
        
      // Compute the element-specific data for the current element.
      // This involves computing the location of the quadrature points and
      // the shape functions (phi, dphi) for the current element.
      fe_vel->reinit  (elem);
      fe_pres->reinit (elem);
        
      
      // Zero the element matrix and right-hand side before summing them.
      // We use the resize member here because the number of DOF might have changed from
      // the last element. Note that this will be the case if the element type is different
      // (i.e. the last element was a triangle, now we are on a quadrilateral).
      Ke.resize (n_dofs, n_dofs);
      Fe.resize (n_dofs);

      // Reposition the submatrices...  The idea is this:
      //
      //         -           -          -  -
      //        | Kuu Kuv Kup |        | Fu |
      //   Ke = | Kvu Kvv Kvp |;  Fe = | Fv |
      //        | Kpu Kpv Kpp |        | Fp |
      //         -           -          -  -
      //
      // DenseSubMatrix.repostition (row_offset, column_offset, row_size, column_size).
      // DenseSubVector.reposition () member takes the (row_offset, row_size)
      Kuu.reposition (u_var*n_u_dofs, u_var*n_u_dofs, n_u_dofs, n_u_dofs);
      Kuv.reposition (u_var*n_u_dofs, v_var*n_u_dofs, n_u_dofs, n_v_dofs);  // 0 matrix
      Kup.reposition (u_var*n_u_dofs, p_var*n_u_dofs, n_u_dofs, n_p_dofs);

      Kvu.reposition (v_var*n_v_dofs, u_var*n_v_dofs, n_v_dofs, n_u_dofs);  // 0 matrix
      Kvv.reposition (v_var*n_v_dofs, v_var*n_v_dofs, n_v_dofs, n_v_dofs);
      Kvp.reposition (v_var*n_v_dofs, p_var*n_v_dofs, n_v_dofs, n_p_dofs);

      Kpu.reposition (p_var*n_u_dofs, u_var*n_u_dofs, n_p_dofs, n_u_dofs);
      Kpv.reposition (p_var*n_u_dofs, v_var*n_u_dofs, n_p_dofs, n_v_dofs);
      Kpp.reposition (p_var*n_u_dofs, p_var*n_u_dofs, n_p_dofs, n_p_dofs);

      Fu.reposition (u_var*n_u_dofs, n_u_dofs);
      Fv.reposition (v_var*n_u_dofs, n_v_dofs);
      Fp.reposition (p_var*n_u_dofs, n_p_dofs);

      // Now we will build the element matrix.
      const Real mu = 1.0;    // kinematic viscosity
      for (unsigned int qp=0; qp<qrule.n_points(); qp++)
      {
        // Assemble the u-velocity row uu coupling
        for (unsigned int i=0; i<n_u_dofs; i++)
          for (unsigned int j=0; j<n_u_dofs; j++)
            Kuu(i,j) += JxW[qp]*(dphi[i][qp]*dphi[j][qp])*mu; // with viscosity

        /// --------------------------------------------------------------------------------
//        if(num_elem==1)
//        {
//          unsigned int i=0, j=0;
//          std::cout << "--------------------- element " << num_elem << std::endl;
//          std::cout << "dphi[i][qp] = " <<dphi[i][qp] << std::endl;
//          std::cout << "dphi[j][qp] = " <<dphi[j][qp] << std::endl;
//          std::cout << "dphi[i][qp]*dphi[j][qp] = " <<dphi[i][qp]*dphi[j][qp] << std::endl;
//          // test above RealGradient multiplication, which is by dot product rule.
//        }
        /// --------------------------------------------------------------------------------
        
        // up coupling
        for (unsigned int i=0; i<n_u_dofs; i++)
          for (unsigned int j=0; j<n_p_dofs; j++)
            Kup(i,j) += -JxW[qp]*psi[j][qp]*dphi[i][qp](0);


        // Assemble the v-velocity row vv coupling
        for (unsigned int i=0; i<n_v_dofs; i++)
          for (unsigned int j=0; j<n_v_dofs; j++)
            Kvv(i,j) += JxW[qp]*(dphi[i][qp]*dphi[j][qp])*mu; // with viscosity

        // vp coupling
        for (unsigned int i=0; i<n_v_dofs; i++)
          for (unsigned int j=0; j<n_p_dofs; j++)
            Kvp(i,j) += -JxW[qp]*psi[j][qp]*dphi[i][qp](1);


        // Assemble the pressure row pu coupling
        for (unsigned int i=0; i<n_p_dofs; i++)
          for (unsigned int j=0; j<n_u_dofs; j++)
            Kpu(i,j) += -JxW[qp]*psi[i][qp]*dphi[j][qp](0);

        // pv coupling
        for (unsigned int i=0; i<n_p_dofs; i++)
          for (unsigned int j=0; j<n_v_dofs; j++)
            Kpv(i,j) += -JxW[qp]*psi[i][qp]*dphi[j][qp](1);

      } // end of the quadrature point qp-loop
      
      
      // ---------------------------------------------------------------------------------------
      // --------------------------------- add the point force ---------------------------------
      // first examine if this element contains particle
      bool contain_p = polymer_chain_two_beads.contain_particle(num_elem);
      if( contain_p )
      {
        // output nodal coordinates in this element
        std::cout<<"------- print info at assemble_stokes_() when add the point force -------"<<std::endl;
        std::cout<<"element "<<num_elem<<" contains particles, and its node coords are: "<<std::endl;
        for(unsigned int i=0; i<n_u_dofs; ++i)
          std::cout<<"node "<<i<<": "<<elem->point(i)<<std::endl;

        // the particle id's
        const std::vector<dof_id_type> particle_ids =
                         polymer_chain_two_beads.element_particle_map(num_elem);
        
        // loop over each particle
        for(unsigned int i=0; i<particle_ids.size(); ++i)
        {
          // the current particle id => its coordinates
          const dof_id_type p_id = particle_ids[i];
          const Point p_xyz = polymer_chain_two_beads.particle_coordinate(p_id);
          
          // point force magnitude
          const Real f0 = 2.0;
          Point p_force = polymer_chain_two_beads.bead_force(p_id);
          const Real fx = f0*p_force(0);
          const Real fy = f0*p_force(1);
          
          // compute the shape functions at this point
          Real N_sum = 0.0;
          for(unsigned int j=0; j<n_u_dofs; ++j)
          {
            Real Nj = FE<2,LAGRANGE>::shape(elem,SECOND,j,p_xyz); // 2D-Lagrangian elem
            Fu(j) += Nj*fx;    Fv(j) += Nj*fy;
            std::cout<<"Shape functions at the point "<<p_xyz<<" in the element "
            <<num_elem << " are: "<< Nj << std::endl;
            N_sum += Nj;
          } // end for j-loop
          std::cout<<"The sum of the shape functions is "<< N_sum<<std::endl;
          
        } // end for i-loop
      } // end if( contain_p )
      // --------------------------------- add the point force ---------------------------------
      // ---------------------------------------------------------------------------------------

      // At this point the interior element integration has been completed.
      // However, we have not yet addressed boundary conditions.
      // For this example we will only consider simple Dirichlet boundary conditions
      // imposed via the penalty method.
      
      { // ****************** (This pair of brackets is not necessary !!!)
        // The following loops over the sides of the element. If the element has no
        // neighbor on a side, then side MUST live on a boundary of the domain.
        for (unsigned int s=0; s<elem->n_sides(); s++)
        {
          if (elem->neighbor(s) == NULL)
          {
            AutoPtr<Elem> side (elem->build_side(s));

            // Loop over the nodes on the side.
            for (unsigned int ns=0; ns<side->n_nodes(); ns++)
            {
              // The location on the boundary of the current node.
              const Real xf = side->point(ns)(0);
              const Real yf = side->point(ns)(1);

              // The penalty value.
              const Real penalty = 1.e10;

              // The boundary values:  Set u = 1 on the top boundary, 0 everywhere else
//              const Real u_value = (yf > .99) ? 1. : 0.;
//              const Real v_value = 0.;                          // Set v = 0 everywhere
              
              /// ----------------------------------B.C.-------------------------------------
              /// my new boundary conditions: u=0,v=0 when yf=0 or yf=1 (non-slip)
              //                              u=1,v=0 at left xf=-1 and +1 ???
              Real u_value = 0., v_value = 0.;
              if (xf>0.999 || xf<-0.999)  //{  u_value = 1.0;    }
              {  u_value = GeomTools::quadratic_function(yf);     }  // apply quadratic u
              
              /// ----------------------------------B.C.-------------------------------------

              // Find the node on the element matching this node on the side.
              // That defined where in the element matrix the B.C. will be applied.
              for (unsigned int n=0; n<elem->n_nodes(); n++)
              {
                if (elem->node(n) == side->node(ns)) // compare global id (dof_id_type)
                {
                  // Matrix contribution.
                  Kuu(n,n) += penalty;
                  Kvv(n,n) += penalty;

                  // Right-hand-side contribution.
                  Fu(n) += penalty*u_value;
                  Fv(n) += penalty*v_value;
                }
              } // end for (n)
            } // end face node loop for (ns)
          } // end if (elem->neighbor(side) == NULL)
        } // end for (s)
      } // end boundary condition section *** (This pair of brackets is not necessary !!!)

      // If this assembly program were to be used on an adaptive mesh,
      // we would have to apply any hanging node constraint equations.
      dof_map.constrain_element_matrix_and_vector (Ke, Fe, dof_indices);

      // The element matrix and right-hand-side are now built
      // for this element.  Add them to the global matrix and
      // right-hand-side vector.  The \p NumericMatrix::add_matrix()
      // and \p NumericVector::add_vector() members do this for us.
      system.matrix->add_matrix (Ke, dof_indices);
      system.rhs->add_vector    (Fe, dof_indices);
      num_elem += 1;  // element id
      
      /// ----------------------------------- My output -----------------------------------
      //      if(num_elem == 1)
      //      {
      //        std::cout<<"n_dofs = "<<n_dofs<<", "<<std::endl
      //                 <<"n_u_dofs = "<<n_u_dofs<<", "<<std::endl
      //                 <<"n_v_dofs = "<<n_v_dofs<<", "<<std::endl
      //                 <<"n_p_dofs = "<<n_p_dofs<<", "<<std::endl;
      //        std::cout << "the size of JxW vector is :"<< JxW.size() << std::endl;
      //        std::cout << "the size of dphi matrix is :"<<dphi.size()<< std::endl;
      //        std::cout << "the size of psi matrix is :"<< psi.size() << std::endl;
      //
      //        // output the value of dphi[Ni][qp](xi)
      //        for(unsigned int i=0; i<dphi.size(); ++i)
      //        {
      //          std::cout << "dphi[" << i << "][0] = " << dphi[i][0] <<std::endl;
      //        }
      //
      //        // output the values of psi[Ni][qp]
      //        for(unsigned int i=0; i<psi.size(); ++i) // 4 shape fun for p, and 9 gauss pts
      //        {
      //          std::cout << "psi[" << i << "] = " ;
      //          for(unsigned int j=0; j<qrule.n_points(); ++j)
      //            std::cout << psi[i][j] <<", ";
      //          std::cout << std::endl;
      //        }
      //
      //        // Prints relevant information about the element.
      //        elem->print_info();
      //
      //        // output nodal coordinates in an element and compare with above print_info
      //        for(unsigned int i=0; i<n_u_dofs; ++i)
      //        {
      //          std::cout<<"node "<<i<<": "<<elem->point(i)<<std::endl;
      //        }
      //      }
      /// ----------------------------------- My output -----------------------------------
      
    } // end of element loop
    
    /// --------------------------- My output ---------------------------
    std::cout <<"total number of elements is :"<< num_elem <<std::endl;

  // That's it.
  return;
}
