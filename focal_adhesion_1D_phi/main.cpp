/*

*******************************************************************************
Copyright (c) 2017-2023 Team hiperlife
Authors: Daniel Santos-Oliván, Alejandro Torres-Sánchez and Guillermo Vilanova
Contributors:
*******************************************************************************
This file is part of hiperlife
Project homepage: https://hiperlife.gitlab.io/hiperlife
Distributed under the GNU General Public License, see the accompanying
file LICENSE or https://opensource.org/license/gpl-3-0.
*******************************************************************************
*/

// C headers

#include <iostream>

// hiperlife headers

#include "hl_Core.h"

#include "hl_Parser.h"

#include "hl_TypeDefs.h"

#include "hl_StructMeshGenerator.h"

#include "hl_DistributedMesh.h"

#include "hl_FillStructure.h"

#include "hl_DOFsHandler.h"

#include "hl_HiPerProblem.h"

#include "hl_LinearSolver_Direct_MUMPS.h"

#include "hl_NonlinearSolver_NewtonRaphson.h"

#include "hl_GlobalBasisFunctions.h"

#include "hl_SurfLagrParam.h"

#include "hl_ConsistencyCheck.h"


#include "hl_MeshLoader.h"

#include <iostream>

/// Trilinos headers
#include <Teuchos_RCP.hpp>


#include <random>
#include <cmath>


static double chi      = -6.5; //==Fix this
static double kon_phi  = 100.0;
static double koff_phi = 10;
static double deltat   = 1E-5;
static double dt_scale_v = 1e2;
static double mobility = 0.068;
static double epsilon2 = 2E-5; //diffusion in the phase field model.
static double x_0 = 5;
static double x_1 = 6;
static double x_m = 5.5;
static int flag_domain_type = 3;   // 1 for biperiodic, 2 for sector and 3 for circle
static double xi_2          =    0.00;
static double theta         = 30.*M_PI/180.;
static double fric          = 0.01;
static double k_b           = 1;
static double K_subs        = 0.5;
static double alpha_1       = 3.0;
static double alpha_2       = 3.; 
static double v_0           = 20;
static double D_u           = 0.3; //=== diffusion of the u-field
static double beta0         = 0.0;
bool flag_micropattern      = 0.0;
bool flag_u_quasistatic     = false;  // true: u = v/alpha(phi) instantly (no drift, no oscillations — diagnostic)


double addGaussianNoise(double mean, double stddev)
{
    std::random_device rd; //Seed for the random number engine
    std::mt19937 generator(rd());   // Mersenne Twister generator
    std::normal_distribution<double> distribution(mean,stddev);
    return distribution(generator); 
}


// double slip_rate(double f, double v0)
// {
//     // Catch-slip parameters (nondimensional, biologically reasonable)
//     const double kc = 1.0;    // catch amplitude
//     const double ks = 0.35;   // slip amplitude (smaller than catch)
//     const double ac = 35.0;   // catch sensitivity (gentler)
//     const double as = 80.0;   // slip sensitivity (steeper)
//     const double fc = 0.05;   // slip activation scale
//     return kc * std::exp(-ac * f) + ks * std::exp(as * (f - fc));
// }


double  slip_rate(double f,double v0)
{
    double f_offset_1 = 0.0;
    double f_offset_2 = 0.05;
    // Scale < 0.5 pushes minimum below 1 (catch regime at f ≈ 0.005)
    double scale = 1;
    double A = scale * (exp(-1000*(f-f_offset_1)) + exp(1000*(f-f_offset_2)));
    return std::min(A, 1e6);
}


double  catch_rate(double f,double v0)
{

    double A = 1; //exp(-3*f);
    if (A<0.5)
        return 0.5;
     return A;
}

void LS(hiperlife::FillStructure& fillStr)
{
    using namespace hiperlife;
    using namespace hiperlife;
    using namespace std;
    using Teuchos::RCP;
    using namespace hiperlife::Tensor;
    using ttl::index::k, ttl::index::l, ttl::index::m, ttl::index::n, ttl::index::o, ttl::index::p, ttl::index::i; 
    using ttl::index::K, ttl::index::L;
    // Rename variables
    SubFillStructure& subFill = fillStr["dhand"];
    int DOF   = subFill.numDOFs;
    int eNN   = subFill.eNN;
    int nDim  = subFill.nDim;
    int pDim  = subFill.pDim;
    int auxF  = subFill.numAuxF;
    double *xe_nodes  = subFill.nborCoords.data();
    double *ue_nodes  = subFill.nborDOFs.data();
    double deltat_       =   fillStr.paramStr->dparam[0];   
    // Shape functions and derivatives at Gauss points
    double jac;
    double* Na = subFill.nborBFs();
    tensor<double,1> lapNa_(eNN);
    tensor<double,2> gNa_(eNN,pDim);
    tensor<double,3> hNa_(eNN,pDim,pDim);

    std::vector<double> gNa(eNN*pDim), hNa(eNN*pDim*pDim), lapNa(eNN);
    GlobalBasisFunctions::hessians(gNa.data(), jac, hNa.data(), lapNa.data(), subFill);
    GlobalBasisFunctions::hessians(gNa_.data(), jac, hNa_.data(), lapNa_.data(), subFill);

    // Values at integration points
    std::vector<double> x(pDim), gphi(pDim), gphin(pDim);
    double phi{}, phin{}, lapphi{},phi2{};
    double v[2]{},u[2]{},u_s[2]{},nn[2]{},q[2]{},gq1[2]{}, gq2[2]{} ;
    Array::Fill(x.data(), pDim, 0.0);
    Array::Fill(gphi.data(), pDim, 0.0);
    Array::Fill(gphin.data(), pDim, 0.0);
    double *ue_nodes0 = subFill.nborDOFs0.data();
    double *aux_nodes = subFill.nborAuxF.data();
    
    for (int i = 0; i < eNN; i++)
    {
        // Values
        for (int d = 0; d < pDim; d++)
            x[d] += xe_nodes[nDim*i+d] * Na[i];
        phi    += ue_nodes[DOF*i+0]  * Na[i];
        phin   += ue_nodes0[DOF*i+0] * Na[i];
        v[0]   += aux_nodes[auxF*i+0]  * Na[i];
        v[1]   += aux_nodes[auxF*i+1] * Na[i];
        u[0]   += aux_nodes[auxF*i+2]  * Na[i];
        u[1]   += aux_nodes[auxF*i+3] * Na[i];
        u_s[0] += aux_nodes[auxF*i+4]  * Na[i];
        u_s[1] += aux_nodes[auxF*i+5] * Na[i];
        // Gradients
        for (int d = 0; d < pDim; d++)
        {
            gphi[d]  += ue_nodes[DOF*i+0] * gNa[pDim*i+d];
            gphin[d] += ue_nodes0[DOF*i+0] * gNa[pDim*i+d];    
        }
        // Hessians
        for (int d = 0; d < pDim; d++)
            lapphi += ue_nodes[i] * hNa[pDim*pDim*i+pDim*d+d];
    }

    double v_mag = sqrt(v[0]*v[0] + v[1]*v[1]);
    double u_mag = sqrt(u[0]*u[0] + u[1]*u[1]);
    double us_mag = sqrt(u_s[0]*u_s[0] + u_s[1]*u_s[1]);
    double fx_ = k_b*(u[0]-u_s[0]);
    double fy_ = k_b*(u[1]-u_s[1]);
    double f_mag = sqrt(fx_*fx_ + fy_*fy_);
    double A = slip_rate(f_mag,1);
    double A_catch = catch_rate(f_mag,1);
    double xi_2_ = xi_2*f_mag/(4.0*A_catch);
    double chi_ = chi/A_catch;

    v_mag = sqrt(v[0]*v[0] + v[1]*v[1]);
    double r    = sqrt(x[0]*x[0] +  x[1]*x[1]);
    // double kon_cutoff = 0.5*(1.0 - tanh((phin - x_m)));
    double mobility_ = mobility;
    double epsilon2_  = epsilon2;

    double phi_0 = x_0;
    double phi_1 = x_1;
    double fa_maturity    = 0.5*(1.0 + tanh(100*(phin - 1.5*x_m)));
    double threshold_mobility = 1.0 - fa_maturity;
    double blend = (1.0 - fa_maturity) * A + fa_maturity;
    double koff_ = koff_phi*A;
    double kon_  = kon_phi;
    mobility_  *= threshold_mobility;
    epsilon2_  *= (threshold_mobility + 1E-3*fa_maturity);
    tensor<double,1> f={0,0};

    // double directional_gradient =  v[0]*gphin[0] +  v[1]*gphin[1]; 
    // double s = (directional_gradient >= 0.0) ? 1.0 : -1.0;

    // only direction, no magnitude
    f(0) = k_b*(u[0]-u_s[0]);
    f(1) = k_b*(u[1]-u_s[1]);

    double alpha_log =0*1E-2;
    double phi_eps = std::max(phi, 1e-12);

    // log lower-bound penalty
    double W_log   = alpha_log * phi_eps * log(phi_eps);
    double dW_log  = alpha_log * (log(phi_eps) + 1.0);
    double ddW_log = alpha_log / phi_eps;
    double dddW_log= -alpha_log / (phi_eps * phi_eps);

    // Double well and derivatives
    double W_sub   = (pow((phi-phi_0),4)  +  pow(phi_1-phi,4)-1)/12.0    + W_log ;
    double dW_sub = (4*pow((phi-phi_0),3)  -  4*pow(phi_1-phi,3))/12.0   + dW_log;
    double ddW_sub = (12*pow((phi-phi_0),2)  +  12*pow(phi_1-phi,2))/12.0+ ddW_log ;
    double dddW_sub= (24*(phi-phi_0)  -  24*(phi_1-phi))/12.0 + dddW_log;

    double W = W_sub + 0.5 * chi_ * (phi-phi_0) * (phi-phi_0) ;
    double dW = dW_sub + chi_ * (phi-phi_0)    ;
    double ddW = ddW_sub  + chi_  ;
    double dddW = dddW_sub  ;
    // Fill
    for (int i = 0; i < eNN; i++)
    {
        // Gradient of basis function times gradient of phase-field variableñ
        double gNagphi{};
        for (int d = 0; d < pDim; d++)
            gNagphi += gNa[pDim*i+d]*gphi[d];

        for (int j = 0; j < eNN; j++)
        {
            // Product of the gradient of basis functions
            double gNagNb{};
            double  gNb{};
            for (int d = 0; d < pDim; d++)
            {
                gNagNb += gNa[pDim*i+d]*gNa[pDim*j+d];
            }
            // Fill Ak
            fillStr.Ak(0, 0)[i*DOF*eNN+j*DOF] += jac * Na[i] * (Na[j] * (1./deltat_)   );
            fillStr.Ak(0, 0)[i*DOF*eNN+j*DOF] += jac * Na[i] * Na[j] * (kon_ + koff_);
            fillStr.Ak(0, 0)[i*DOF*eNN+j*DOF] += jac * mobility_ * (ddW + phi * dddW) * Na[j] * gNagphi;
            fillStr.Ak(0, 0)[i*DOF*eNN+j*DOF] += jac * mobility_ * phi * ddW * gNagNb;
            fillStr.Ak(0, 0)[i*DOF*eNN+j*DOF] += jac * epsilon2_ * ((phi * lapNa[i] + gNagphi) * lapNa[j] + (lapNa[i] * Na[j] + gNagNb) * lapphi);

        }
        // Fill RHS
        fillStr.Bk(0)[i*DOF] += jac * ( Na[i] * ((phi-phin) / deltat_   - kon_*std::max(x_m-phi,0.) + koff_ * phi ) + mobility_ * (phi * ddW * gNagphi) + epsilon2_ * (phi * lapNa[i] + gNagphi) * lapphi     );

    }


    wrapper<double,2> Bk(fillStr.Bk(0).data(),eNN,DOF);
    wrapper<double,4> Ak(fillStr.Ak(0,0).data(),eNN,DOF,eNN,DOF);
    wrapper<double,2> nborCoords(subFill.nborCoords.data(),eNN,nDim);
    wrapper<double,1> bf_(subFill.getDer(0),eNN);
    wrapper<double,2> Dbf_l(subFill.getDer(1),eNN,pDim);

    tensor<double,2> T = nborCoords(all,range(0,1)).T() * Dbf_l;
    //Global derivatives (wrt x and y )  of the basis functions
    tensor<double,2>  Dbf_g = Dbf_l*T.inv();
    constexpr double threshold = 1e-6;

    const double f_mag2 = f * f;

    tensor<double, 2> P ={{0, 0 },{0, 0}};  // zero-initialized

    if (f_mag2 >= threshold * threshold) {
        P = outer(f, f) / f_mag2;
}

    tensor<double,1> gphi_(2); 
    gphi_(0)  = gphi[0]; 
    gphi_(1)  = gphi[1];

    Bk(all ,0)(K)                  +=  jac*xi_2_*P(i,k)*(phi * ddW*Dbf_g(K,i)*gphi_(k)); 
    Ak(all ,0,all,0)(K,L)          +=  jac*xi_2_*P(i,k)*(phi * ddW*Dbf_g(K,i)*Dbf_g(L,k)  + phi * dddW*bf_(L)*Dbf_g(K,i)*gphi_(k) + bf_(L)*ddW*Dbf_g(K,i)*gphi_(k)); 

    Bk(all ,0)(K)                  +=  jac* epsilon2_*( P(i,k)*gphi_(i)*Dbf_g(K,k)     ) ;
    Ak(all ,0,all,0)(K,L)          +=  jac* epsilon2_*( P(i,k)*Dbf_g(L,i)*Dbf_g(K,k)   ) ;
 
}



void LS_border(hiperlife::FillStructure& fillStr)
{
    using namespace std;
    using namespace hiperlife;

    // Variables
    SubFillStructure& subFill = fillStr["dofHand"];
    int DOF  = subFill.numDOFs;                      // Number of degrees of freedom
    int eNN  = subFill.eNN;                          // Number of nodes of the neighbophiod
    int nDim = subFill.nDim;                         // Dimension of the space of the problem
    int pDim = subFill.pDim;                         // Dimension of the space of the problem
    double *xe_nodes  = subFill.nborCoords.data();
    double *ue_nodes  = subFill.nborDOFs.data();
    double *ue_nodes0 = subFill.nborDOFs0.data();

    double jac;
    double* Na = subFill.nborBFs();
    std::vector<double> gNa(eNN*pDim), hNa(eNN*pDim*pDim), lapNa(eNN);
    GlobalBasisFunctions::hessians(gNa.data(), jac, hNa.data(), lapNa.data(), subFill);

    // Compute Tangent Map
    double Ip[4], xu[3], xv[3];
    SurfLagrParam::MetricTensor(Ip, xu, xv, eNN, xe_nodes, subFill.nborBasisFunctionsGradients());

    // Compute jacobian
    auto tangentRef = subFill.tangentsBoundaryRef();
    double tangent[3]={};
    for (int n = 0; n < nDim; n++)
        tangent[n] = tangentRef[0] * xu[n] + tangentRef[1]*xv[n];
    jac = Math::Norm3D(tangent);
    double normal[2] = {tangent[1],-tangent[0]}; 

    // Values at integration points
    std::vector<double> x(pDim), gphi(pDim);
    double phi{}, phin{}, lapphi{};
    Array::Fill(x.data(), pDim, 0.0);
    Array::Fill(gphi.data(), pDim, 0.0);
    for (int i = 0; i < eNN; i++)
    {
        // Values
        for (int d = 0; d < pDim; d++)
            x[d] += xe_nodes[nDim*i+d] * Na[i];
        phi  += ue_nodes[DOF*i+0]  * Na[i];
        phin += ue_nodes0[DOF*i+0] * Na[i];

        // Gradients
        for (int d = 0; d < pDim; d++)
            gphi[d] += ue_nodes[DOF*i+0] * gNa[pDim*i+d];

        // Hessians
        for (int d = 0; d < pDim; d++)
            lapphi += ue_nodes[i] * hNa[pDim*pDim*i+pDim*d+d];
    }

    double gphiNormal{};
    for (int d = 0; d < pDim; d++)
        gphiNormal += gphi[d] * normal[d];

    // Fill
    for (int i = 0; i < eNN; i++)
    {
        // Gradient of basis function times gradient of phase-field variable
        double gNagphi{}, gNaNormal{};
        for (int d = 0; d < pDim; d++)
        {
            gNagphi += gNa[pDim*i+d]*gphi[d];
            gNaNormal += gNa[pDim*i+d]*normal[d];
        }    
        

        for (int j = 0; j < eNN; j++)
        {
            // Product of the gradient of basis functions
            double gNagNb{}, gNbNormal{};
            for (int d = 0; d < pDim; d++)
            {
                gNagNb += gNa[pDim*i+d]*gNa[pDim*j+d];
                gNbNormal += gNa[pDim*j+d]*normal[d];
            }
            // Fill Ak
            // fillStr.Ak(0, 0)[i*eNN+j] -= jac * mobility * epsilon2 * (gNaNormal * (Na[j] * lapphi + phi * lapNa[j]));
            // fillStr.Ak(0, 0)[i*eNN+j] -= jac * mobility * epsilon2 * ((lapNa[i] * Na[j] + Na[i] * lapNa[j]) * gphiNormal + (lapNa[i] * phi + Na[i] * lapphi) * gNbNormal);
            // // fillStr.Ak(0, 0)[i*eNN+j] += jac * mobility * epsilon2 * lapNa[i] * (phi * gNbNormal + Na[j] * gphiNormal);
            fillStr.Ak(0, 0)[i*eNN+j] += jac * 1E6 * gNbNormal * gNaNormal;
        }
        // Fill RHS
        // fillStr.Bk(0)[i] -= jac * mobility * epsilon2 * (phi * lapphi * gNaNormal);
        // fillStr.Bk(0)[i] -= jac * mobility * epsilon2 * (lapNa[i] * phi + Na[i] * lapphi) * gphiNormal;
        // fillStr.Bk(0)[i] += jac * mobility * epsilon2 * lapNa[i] * phi * gphiNormal;
        fillStr.Bk(0)[i] += jac * 1E6 * gphiNormal * gNaNormal; 

    }
}


void LS_v(hiperlife::FillStructure& fillStr)
{
    using namespace hiperlife;
    using namespace hiperlife;
    using namespace std;
    using Teuchos::RCP;
    using namespace hiperlife::Tensor;
    using ttl::index::k, ttl::index::l, ttl::index::m, ttl::index::n, ttl::index::o, ttl::index::p; 
    using ttl::index::K, ttl::index::L;
    // Rename variables
    SubFillStructure& subFill = fillStr["dhand"];
    int DOF   = subFill.numDOFs;
    int eNN   = subFill.eNN;
    int nDim  = subFill.nDim;
    int pDim  = subFill.pDim;
    int auxF  = subFill.numAuxF;
    double gamma  = 20.0;
    double visc   = 1;
    double fric_ = fric;
    double tau  = 1;
    double tau_2 = 1;
    // double gamma  = 0.0109/5.5;
    // double visc   = 0.001;
    tensor<double,1> v_star = {v_0,0};
    double deltat_       =   dt_scale_v*fillStr.paramStr->dparam[0];

    // Rename variables for the main dofsHandler
    int numAuxF = subFill.numAuxF;
    wrapper<double,2> nborCoords(subFill.nborCoords.data(), eNN, nDim);
    wrapper<double,2> nborDOFs(subFill.nborDOFs.data(), eNN, DOF);
    wrapper<double,2> nborDOFs0(subFill.nborDOFs0.data(), eNN, DOF);
    wrapper<double,2> nborauxF(subFill.nborAuxF.data(), eNN, auxF);
    wrapper<double,1> bf(subFill.nborBFs(), eNN);
    wrapper<double,2> Dbf_l(subFill.getDer(1),eNN,pDim);
    tensor<double,2>  id2d = Identity(2);

    // Get output
    wrapper<double,2> Bk(fillStr.Bk(0).data(),eNN,DOF);
    wrapper<double,4> Ak(fillStr.Ak(0,0).data(),eNN,DOF,eNN,DOF);


    tensor<double,2> T       =  (nborCoords(all,range(0,1))(L,k)*Dbf_l(L,l))(k,l);
    tensor<double,2> Tinv    =  T.inv();   

    tensor<double,2> Dbf_g      =  Dbf_l(K,k)*Tinv(k,l);
    tensor<double,2> nbor_v      = nborDOFs(all,range(0,1));
    tensor<double,2> nbor0_v    = nborDOFs0(all,range(0,1));
    tensor<double,1> nbor_phi   = nborauxF(all,0);
    tensor<double,1> v          = bf(K) * nbor_v(K,k);
    tensor<double,1> v0         = bf(K) * nbor0_v(K,k);

    tensor<double,2> nbor_us    = nborDOFs(all,range(2,3));
    tensor<double,2> nbor0_us   = nborDOFs0(all,range(2,3));
    tensor<double,1> us         = bf(K)* nbor_us(K,k);
    tensor<double,1> us_0       = bf(K)* nbor0_us(K,k);

    // u is now a DOF (indices 4,5) — solved by FEM, not by pointwise loop
    tensor<double,2> nbor_u     = nborDOFs(all,range(4,5));
    tensor<double,2> nbor0_u    = nborDOFs0(all,range(4,5));
    tensor<double,1> u          = bf(K)* nbor_u(K,k);
    tensor<double,1> u0         = bf(K)* nbor0_u(K,k);

    double phi = bf(K) * nbor_phi(K);

    tensor<double,2> Dv          =  (nbor_v(K,m)*Dbf_g(K,n))(m,n);

    double jac = T.det();

    tensor<double,4> dvDv = (Dbf_g(K,n)*id2d(m,k))(K,k,m,n) ; 
    tensor<double,2> d_divv  =  (dvDv(K,k,l,m)*id2d(l,m))(K,k);

    tensor<double,2> rodt     = (0.5*(Dv(m,n)  + Dv(n,m)))(m,n);
    double divv = rodt(0,0) + rodt(1,1)  ;
    tensor<double,2> d_divu  =  (dvDv(K,k,l,m)*id2d(l,m))(K,k);
    tensor<double,4> dv_rodt  = (0.5*(dvDv(K,k,m,n)  + dvDv(K,k,n,m)))(K,k,m,n);

    //#######################################################################################//   
    //###############################  ACTIN CORTEX EQUATIONS ##############################//
    //#####################################################################################//


    // //[9.2] GRADIENT
    Bk(all,range(0,1))(K,k) +=  gamma*jac*d_divv(K,k);
    Bk(all,range(0,1))(K,k)           +=  jac * visc * (dv_rodt(K,k,m,n)*rodt(m,n))(K,k);
    // Bk(all,range(0,1))(K,k)           +=  jac * visc * divv*d_divv(K,k);
    // //   //[12.2] HESSIAN
    Ak(all,range(0,1),all,range(0,1))(K,k,L,l) +=  jac * visc * dv_rodt(K,k,m,n)*dv_rodt(L,l,m,n);
    // Ak(all,range(0,1),all,range(0,1))(K,k,L,l) +=  jac * visc * d_divv(L,l)*d_divv(K,k);

    //#######################################################################################//
    //###############################  FRICTION FORCE FROM ENV #############################//
    //#####################################################################################//
    Bk(all,range(0,1))(K,k)                            +=    fric_*jac* bf(K)*v(k);
    Ak(all ,range(0,1),all ,range(0,1))(K,k,L,l)       +=    fric_*jac* bf(K)*bf(L)*id2d(k,l);

    //#######################################################################################//
    //###############################  BOND FORCE ON ACTIN  ################################//
    // ∂(k_b|u|²)/∂u = k_b·u  →  body force on actin from stretched integrin bonds
    //#####################################################################################//
    Bk(all,range(0,1))(K,k)                            +=   phi*jac* bf(K) * k_b * u(k)/(deltat_*x_m);
    Ak(all,range(0,1),all,range(4,5))(K,k,L,l)         +=   phi*jac* bf(K) * bf(L) * k_b * id2d(k,l)/(deltat_*x_m);
    //#######################################################################################//
    //###############################  ADHESION DISPLACEMENT u  ############################//
    // du/dt = v - alpha_detach*u   (backward Euler, phi timestep)
    //#####################################################################################//
    double deltat_u     = fillStr.paramStr->dparam[0];
    double phi_arrest   = x_m;  // separate from x_m: only mature FAs (phi > x_1) suppress alpha_detach
    double alpha_detach = 1E-6 + 1000*kon_phi * std::max(phi_arrest - phi, 0.0)/std::max(phi, 1e-6);
    //double alpha_detach = 1E-6 + 1000*(1.0 - tanh(10000 * (phi - 4.0)));
    // beta_reset: forces u→0 outside FAs (phi near x_0), zero inside FA (phi near x_1)
    double phi_reset   = x_0 + 0.45*(x_1 - x_0);   // = 5.45, just below x_m=5.5
    double width_reset = 0.05*(x_1 - x_0);          // = 0.05
    double blend_u     = 0.5*(1.0 + tanh((phi - phi_reset)/width_reset));
    double beta_reset  = beta0 * (1.0 - blend_u);

    Bk(all,range(4,5))(K,k) += jac * bf(K) * ((u(k) - u0(k))/deltat_u + (alpha_detach + beta_reset)*u(k) - v(k));
    Ak(all,range(4,5),all,range(4,5))(K,k,L,l) += jac * bf(K) * bf(L) * (1.0/deltat_u + alpha_detach + beta_reset) * id2d(k,l);
    Ak(all,range(4,5),all,range(0,1))(K,k,L,l) -= jac * bf(K) * bf(L) * id2d(k,l);

    // Diffusion of u: D_u * ∫ ∇N_K · ∇u dΩ
    tensor<double,2> Du = (nbor_u(K,k)*Dbf_g(K,n))(k,n);
    Bk(all,range(4,5))(K,k) += jac * D_u * (Dbf_g(K,n) * Du(k,n));
    Ak(all,range(4,5),all,range(4,5))(K,k,L,l) += jac * D_u * Dbf_g(K,n) * Dbf_g(L,n) * id2d(k,l);

    // M(φ)·∇μ·∇u advection  — full ∇μ = W''∇φ − ε²∇(∇²φ)
    // PDE: du/dt = −M∇μ·∇u + …  →  residual: +∫ N·M·∇μ·∇u dΩ
    double fa_mat_v         = 0.5*(1.0 + std::tanh(100.0*(phi - 1.5*x_m)));
    double mob_v            = mobility * (1.0 - fa_mat_v);
    double tanh_v           = std::tanh(100.0*(phi - 1.5*x_m));
    double dM_dphi_v        = -mobility * 50.0 * (1.0 - tanh_v*tanh_v);   // dM/dφ
    double ddW_v            = (12.0*std::pow(phi-x_0,2) + 12.0*std::pow(x_1-phi,2)) / 12.0 + chi;
    tensor<double,1> gphi_v = Dbf_g(K,n) * nbor_phi(K);   // ∇φ at Gauss point

    // W'' part: +∫ N·M·W''·∇φ·∇u dΩ
    Bk(all,range(4,5))(K,k)            += jac * bf(K) * mob_v * ddW_v * (gphi_v(n) * Du(k,n));
    Ak(all,range(4,5),all,range(4,5))(K,k,L,l) += jac * bf(K) * mob_v * ddW_v * (gphi_v(n) * Dbf_g(L,n)) * id2d(k,l);

    // ε² part after IBP: -∫ N·M·ε²·∇(∇²φ)·∇u  →  +∫ ε²·∇²φ·[M·∇N·∇u + N·dM/dφ·∇φ·∇u + N·M·∇²u]
    {
        tensor<double,1> lapNa_v(eNN);
        tensor<double,2> gNa_tmp(eNN, pDim);
        tensor<double,3> hNa_tmp(eNN, pDim, pDim);
        double jac_tmp;
        GlobalBasisFunctions::hessians(gNa_tmp.data(), jac_tmp, hNa_tmp.data(), lapNa_v.data(), subFill);

        double            laphi_v = lapNa_v(K) * nbor_phi(K);          // ∇²φ
        tensor<double,1>  lapu_v  = lapNa_v(K) * nbor_u(K,k);          // ∇²u_k

        Bk(all,range(4,5))(K,k) += jac * epsilon2 * laphi_v * (
              mob_v    * (Dbf_g(K,n) * Du(k,n))
            + bf(K) * dM_dphi_v * (gphi_v(n) * Du(k,n))
            + bf(K) * mob_v     * lapu_v(k)
        );
        Ak(all,range(4,5),all,range(4,5))(K,k,L,l) += jac * epsilon2 * laphi_v * (
              mob_v    * (Dbf_g(K,n) * Dbf_g(L,n)) * id2d(k,l)
            + bf(K) * dM_dphi_v * (gphi_v(n) * Dbf_g(L,n)) * id2d(k,l)
            + bf(K) * mob_v     * lapNa_v(L)        * id2d(k,l)
        );
    }

}




int main(int argc, char *argv[])
{
    using namespace std;
    using namespace hiperlife;

    // Initialize MPI
    hiperlife::Init(argc, argv);
    const int myRank = hiperlife::MyRank();

    // Mesh-related parameters
    BasisFuncType bfType = BasisFuncType::BSplines; 
    ElemType eType = ElemType::Square;
    int bfOrder = 2;
    int gPts = pow(bfOrder+1, computePDim(eType));
    int nEl = 50;

    // Time-related parameters
    string oname   = "integrin.";
    string oname_v = "velocity.";
    double maxTime  = 10000000;
    double maxNStep = 10000000;
    double maxDelt  = 1E-2;
    double stepFactor = 0.9;
    int nSave = 5;
    double width = 0.1; 
    double L = 0.3;

    SmartPtr<ParamStructure> userStr   = Create<ParamStructure>() ;
    userStr->dparam.resize(50);
    userStr->dparam[0] = deltat;
    // Generate mesh
    SmartPtr<StructMeshGenerator> structMesh = Create<StructMeshGenerator>();
    try
    {
        structMesh->setMesh(eType, bfType, 2);
        structMesh->setPeriodicBoundaryCondition({Axis::Xaxis,Axis::Yaxis});
        structMesh->genRectangle(nEl,nEl, L, L);
    }
    catch (runtime_error& err)
    {
        throw runtime_error("Mesh could not be generated.");
    }

    Teuchos::RCP<MeshLoader> loadedMesh = Teuchos::rcp(new MeshLoader);
    loadedMesh->setElemType(ElemType::Triang);
    loadedMesh->setBasisFuncType(BasisFuncType::SubdivSurfs);
    loadedMesh->setBasisFuncOrder(bfOrder);
    loadedMesh->loadMesh("/home/mirza/src/apps/endothelium_modelling/circle_n2.vtk",hiperlife::MeshType::Parallel);  

    // Create distributed mesh
    SmartPtr<DistributedMesh> disMesh = Create<DistributedMesh>();
    try
    {
        if (flag_micropattern)
            disMesh->setMesh(loadedMesh);
        else
            disMesh->setMesh(structMesh);         
        disMesh->Update();

        cout << disMesh->myRank() << ":   Distributed Mesh successfully created. " << endl;
    }
    catch (runtime_error& err)
    {
        cout << disMesh->myRank() << ":   Distributed Mesh could not be created. " << err.what() << endl;

        MPI_Finalize();
        return 0;
    }

    // Print mesh
    disMesh->printFileVtk("Mesh");

    // DOFsHandler
    SmartPtr<DOFsHandler> dhand = Create<DOFsHandler>(disMesh);
    SmartPtr<DOFsHandler> dhand_v = Create<DOFsHandler>(disMesh);
    try
    {
        dhand->setNameTag("dhand");
        dhand->setDOFs({"phi"});
        dhand->setNodeAuxF({"vx","vy","ux","uy","us_x","us_y"});
        dhand->Update();
        cout << dhand->myRank() << ":   DOFsHandler successfully created. " << endl;
    }
    catch (runtime_error& err)
    {
        cout << dhand->myRank() << ":   DOFsHandler could not be created. " << err.what() << endl;

        MPI_Finalize();
        return 0;
    }

    try
    {
        dhand_v->setNameTag("dhand_v");
        dhand_v->setDOFs({"vx","vy","us_x","us_y","ux","uy"});
        dhand_v->setNodeAuxF({"phi"});
        dhand_v->Update();
        cout << dhand->myRank() << ":   DOFsHandler successfully created. " << endl;
    }
    catch (runtime_error& err)
    {
        cout << dhand->myRank() << ":   DOFsHandler could not be created. " << err.what() << endl;

        MPI_Finalize();
        return 0;
    }

    bool flag = true;
    for (int i = 0; i < dhand->mesh->loc_nPts(); i++)
    {

        double x = disMesh->nodeCoord(i, 0, IndexType::Local);
        double y = disMesh->nodeCoord(i, 1, IndexType::Local);

        // Calculate the radius (distance from the origin)
        double radius = std::sqrt(x * x + y * y);

        // Calculate the angle (azimuth) using std::atan2
        double angle = std::atan2(y, x);

        // Scale the angle to have 5 cycles in 2*PI (360 degrees)
        const int NUM_CYCLES = 5; // Number of peaks and troughs
        double scaled_angle = NUM_CYCLES * angle;

        // Calculate sin(scaled_angle)
        double sin_scaled_angle;
        if (radius == 0.0) 
        {
            sin_scaled_angle = 0.0;
        } else 
        {
            sin_scaled_angle = std::sin(scaled_angle);
        }

        // Now using sin_scaled_angle in direction calculation
        double direction = sin_scaled_angle;

        // Parameters for the sine wave
        const double AMPLITUDE = 0.03;      // Maximum deviation from zero
        const double WAVELENGTH = 1.0;     // Spatial wavelength (adjust as needed)
        const double PHASE_OFFSET = 0.0;   // Initial phase shift (optional)


        // Calculate the phase of the sine wave
        double phase = 2.0 * 3.14159 * direction / WAVELENGTH + PHASE_OFFSET;
        double mean  = kon_phi*x_m/(kon_phi + koff_phi * slip_rate(0.0, 0.0));  // phi_ss at f=0
        // Calculate the value of r using the sine wave
        double r  = AMPLITUDE * std::sin(phase) ;
        double r1 = 0.001 * (((double) rand() / (RAND_MAX)) * 2.0 - 1.0);
        double r2 = addGaussianNoise(mean, 0.01*mean);
        // Set the initial value with the sine wave perturbation

        dhand->nodeDOFs->setValue(0, i, IndexType::Local,   r2);
        int crease = dhand_v->mesh->nodeCrease(i, IndexType::Local);
        if (x > L - 1E-2)
        {
            // dhand_v->nodeDOFs->setValue("vx", i, IndexType::Local, 0);
            // dhand_v->setConstraint("vx", i, IndexType::Local, 0);
        }
    }
    if (flag_micropattern)
    {
        for (int i = 0; i < dhand->mesh->loc_nPts(); ++i)
        {
            double x = disMesh->nodeCoord(i, 0, IndexType::Local);
            double y = disMesh->nodeCoord(i, 1, IndexType::Local);
            double r = std::sqrt(x*x + y*y);
            double d = 1 - r;  // distance from boundary
            double vmag = v_0 * std::exp(-d / 0.5);

            double vx = vmag * x / r;
            double vy = vmag * y / r;

            if (r>0.4)
            {
                dhand_v->nodeDOFs->setValue("vx", i, IndexType::Local,   vx);
                dhand_v->nodeDOFs->setValue("vy", i, IndexType::Local,   vy);
            }
            else
            {
                dhand_v->nodeDOFs->setValue("vx", i, IndexType::Local,   0);
                dhand_v->nodeDOFs->setValue("vy", i, IndexType::Local,   0);                
            }

        }
    }
    else
    {
        dhand_v->nodeDOFs->setValue("vx", v_0);
        dhand_v->nodeDOFs->setValue("vy", 0); 
        dhand_v->setConstraint("vx",0); 
        dhand_v->setConstraint("vy",0); 
    }

    dhand_v->nodeDOFs->setValue("ux",0);
    dhand_v->nodeDOFs->setValue("uy",0);
    dhand_v->nodeDOFs->setValue("us_x",0);
    dhand_v->nodeDOFs->setValue("us_y",0);
    dhand_v->setConstraint("us_x",0);
    dhand_v->setConstraint("us_y",0);
    dhand->UpdateGhosts();
    dhand->nodeDOFs0->setValue(dhand->nodeDOFs);
    dhand_v->nodeDOFs0->setValue(dhand_v->nodeDOFs);

    // Create HiPerProblem
    SmartPtr<HiPerProblem> hiperProbl = Create<HiPerProblem>();
    try
    {
        // Set DOFHandler
        hiperProbl->setDOFsHandlers({dhand});

        // Set Integration
        hiperProbl->setIntegration("Integ", {"dhand"});
        if (flag_micropattern)
            hiperProbl->setCubatureGauss("Integ", 3);
        else
            hiperProbl->setCubatureGauss("Integ", gPts); 
       
        hiperProbl->setElementFillings("Integ", LS);
        hiperProbl->setParameterStructure(userStr);
        hiperProbl->setIntegration("BorderInteg", {"dofHand"});
        if (flag_micropattern)
            hiperProbl->setCubatureBorderGauss("BorderInteg",2);
        else
            hiperProbl->setCubatureBorderGauss("BorderInteg",gPts);        
        hiperProbl->setElementFillings("BorderInteg", LS_border);


        // Set global integral

        hiperProbl->setConsistencyCheckDelta(1.E-4);
        hiperProbl->setConsistencyCheckTolerance(1.E-4);
        hiperProbl->setConsistencyCheckType(ConsistencyCheckType::Hessian);
        
        // Update
        hiperProbl->Update();
    }
    catch (runtime_error& err)
    {
        cout << myRank << ": HiPerProblem could not be created " << err.what() << endl;
        MPI_Finalize();
        return 1;
    }


    SmartPtr<HiPerProblem> hiperProbl_v = Create<HiPerProblem>();
    try
    {
        // Set DOFHandler
        hiperProbl_v->setDOFsHandlers({dhand_v});

        // Set Integration
        hiperProbl_v->setIntegration("Integ", {"dhand"});
        if (flag_micropattern)
            hiperProbl_v->setCubatureGauss("Integ", 3);
        else
            hiperProbl_v->setCubatureGauss("Integ", 4); 
        hiperProbl_v->setParameterStructure(userStr);
        // hiperProbl->setElementFillings("Integ", ConsistencyCheck<LS>);
        hiperProbl_v->setElementFillings("Integ", LS_v);
        hiperProbl_v->setConsistencyCheckDelta(1.E-4);
        hiperProbl_v->setConsistencyCheckTolerance(1.E-4);
        hiperProbl_v->setConsistencyCheckType(ConsistencyCheckType::Hessian);
        
        // Update
        hiperProbl_v->Update();
    }   
    catch (runtime_error& err)
    {
        cout << myRank << ": HiPerProblem could not be created " << err.what() << endl;
        MPI_Finalize();
        return 1;
    }

    // Create linear solver
    SmartPtr<MUMPSDirectLinearSolver> linSolver = Create<MUMPSDirectLinearSolver>();
    linSolver->setHiPerProblem(hiperProbl);
    linSolver->setDefaultParameters();
    linSolver->setWorkSpaceMemoryIncrease(200);
    linSolver->Update();


    // Create linear solver
    SmartPtr<MUMPSDirectLinearSolver> linSolver_v = Create<MUMPSDirectLinearSolver>();
    linSolver_v->setHiPerProblem(hiperProbl_v);
    linSolver_v->setDefaultParameters();
    linSolver_v->setWorkSpaceMemoryIncrease(200);
    linSolver_v->Update();


    // Create nonlinear solver
    SmartPtr<NewtonRaphsonNonlinearSolver> nonlinSolver = Create<NewtonRaphsonNonlinearSolver>();
    nonlinSolver->setLinearSolver(linSolver);
    nonlinSolver->setMaxNumIterations(6);
    nonlinSolver->setResTolerance(1.E-6);
    nonlinSolver->setSolTolerance(1.E-6);
    // nonlinSolver->setResMaximum(1.E5);
    // nonlinSolver->setSolMaximum(1.E5);
    nonlinSolver->setLineSearch(false);
    nonlinSolver->setConvRelTolerance(false);
    nonlinSolver->setPrintIntermInfo(true);
    nonlinSolver->setPrintSummary(false);
    nonlinSolver->Update();

    // Create nonlinear solver
    SmartPtr<NewtonRaphsonNonlinearSolver> nonlinSolver_v = Create<NewtonRaphsonNonlinearSolver>();
    nonlinSolver_v->setLinearSolver(linSolver_v);
    nonlinSolver_v->setMaxNumIterations(6);
    nonlinSolver_v->setResTolerance(1.E-6);
    nonlinSolver_v->setSolTolerance(1.E-6);
    // nonlinSolver_v->setResMaximum(1.E8);
    // nonlinSolver_v->setSolMaximum(1.E8);
    nonlinSolver_v->setLineSearch(false);
    nonlinSolver_v->setConvRelTolerance(false);
    nonlinSolver_v->setPrintIntermInfo(true);
    nonlinSolver_v->setPrintSummary(false);
    nonlinSolver_v->Update();


    // Print initial condition
    dhand->nodeDOFs->setValue(dhand->nodeDOFs0);
    dhand->UpdateGhosts();
    dhand->printFileVtk(oname + to_string(0), true, 0.0);

    dhand_v->nodeDOFs->setValue(dhand_v->nodeDOFs0);
    dhand_v->UpdateGhosts();
    dhand_v->printFileVtk(oname + to_string(0), true, 0.0);

    // Open file to write global integrals and write headers
    ofstream gIntegFile;
    gIntegFile.open ("globalIntegrals.csv");
    gIntegFile << "TS time deltat";
    for (auto g: hiperProbl->globalIntegralNames())
        gIntegFile << " " << g;
    gIntegFile << endl;

    // Loop over time steps
    double time{0.0};
    int timeStep{0};
    double ener_prev{1.E8};
    std::ofstream outFile("output.txt");
    outFile << "time" << "," << "vx" << "," << "ux" << std::endl;
    while ((timeStep < maxNStep) and (time < maxTime))
    {
        // Write
        if (dhand->myRank() == 0)
            cout << endl << " TS: " << to_string(timeStep) << " Time " << time <<  " Delta " << deltat << endl;

        // Initial guess
        dhand->nodeDOFs->setValue(dhand->nodeDOFs0);
        hiperProbl->UpdateGhosts();

        // Newton-Raphson method
        int    ierr  = nonlinSolver->solve();

        // cout << dhand->myRank() << " " << min << " " << max << endl;
        // Check convergence
        if (ierr != 0)// and max < 1.0 and min > 0.0)
        {


            // Save solution
            dhand->nodeDOFs0->setValue(dhand->nodeDOFs);

            // Initial guess
            dhand_v->nodeDOFs->setValue(dhand_v->nodeDOFs0);
            hiperProbl->UpdateGhosts();

            // Newton-Raphson method
            int ierr_v = nonlinSolver_v->solve();

            if (ierr_v != 0)
            {

                // u is now solved by LS_v as DOFs — just copy to dhand auxF for use in phi equation
                for (int i = 0; i < dhand->mesh->loc_nPts(); ++i)
                {
                    double vx   = dhand_v->nodeDOFs->getValue("vx",  i, IndexType::Local);
                    double vy   = dhand_v->nodeDOFs->getValue("vy",  i, IndexType::Local);
                    double ux   = dhand_v->nodeDOFs->getValue("ux",  i, IndexType::Local);
                    double uy   = dhand_v->nodeDOFs->getValue("uy",  i, IndexType::Local);
                    double us_x = dhand_v->nodeDOFs->getValue("us_x", i, IndexType::Local);
                    double us_y = dhand_v->nodeDOFs->getValue("us_y", i, IndexType::Local);
                    dhand->nodeAuxF->setValue("vx",   i, IndexType::Local, vx);
                    dhand->nodeAuxF->setValue("vy",   i, IndexType::Local, vy);
                    dhand->nodeAuxF->setValue("ux",   i, IndexType::Local, ux);
                    dhand->nodeAuxF->setValue("uy",   i, IndexType::Local, uy);
                    dhand->nodeAuxF->setValue("us_x", i, IndexType::Local, us_x);
                    dhand->nodeAuxF->setValue("us_y", i, IndexType::Local, us_y);
                }
                dhand->UpdateGhosts();
                dhand_v->UpdateGhosts();
                // Save solution
                dhand_v->nodeDOFs0->setValue(dhand_v->nodeDOFs);
                // Update variables
                timeStep ++;

                if(nonlinSolver->numberOfIterations()<5)
                    deltat /= stepFactor;
                if (deltat >= maxDelt)
                    deltat = maxDelt;
                time += deltat;


                // Write results
                if (timeStep % nSave == 0 and outFile.is_open())
                {
                    string solName   = oname + to_string(timeStep);
                    string solName_v = oname_v + to_string(timeStep);
                    //dhand->printFileVtk(solName, true,time);
                    dhand_v->printFileVtk(solName_v, true,time);
                    double vx  = dhand_v->nodeDOFs->getValue("vx", 0, IndexType::Local);
                    double ux  = dhand_v->nodeDOFs->getValue("ux", 0, IndexType::Local);
                    outFile << time << "," << vx << "," << ux << std::endl;
                }

                for (int i = 0; i < dhand->mesh->loc_nPts(); i++)
                {
                    double phi  = dhand->nodeDOFs->getValue("phi", i, IndexType::Local);
                    dhand_v->nodeAuxF->setValue("phi", i, IndexType::Local, phi);
                }
                dhand_v->UpdateGhosts();

            }
            else
            {

                // Correct time-step size
                deltat *= stepFactor;
                // Save solution
                dhand_v->nodeDOFs->setValue(dhand_v->nodeDOFs0);
                dhand->nodeDOFs->setValue(dhand->nodeDOFs0);
            }

                int iters = std::max(nonlinSolver->numberOfIterations(), nonlinSolver_v->numberOfIterations());
                if (iters > 4)
                    deltat *= stepFactor;
                else
                    deltat /= stepFactor;
        }
        else
        {

            // Correct time-step size
            deltat *= stepFactor;
            dhand_v->nodeDOFs->setValue(dhand_v->nodeDOFs0);
            dhand->nodeDOFs->setValue(dhand->nodeDOFs0);
        }

        userStr->dparam[0] = deltat;
        // Check steady-state
        if (deltat < 1.e-14)
            break;
    }

    // Close file
    gIntegFile.close();

    // Finalize
    hiperlife::Finalize();
    return 0;
}
