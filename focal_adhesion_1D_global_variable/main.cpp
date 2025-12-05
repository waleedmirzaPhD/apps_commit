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


static double chi      = -10;
static double kon  = 2.0;
static double koff = 1;
static double kon_phi  = 100.0;
static double koff_phi = 10;
static double deltat   = 1E-6;
static double mobility = 0.0675;
static double epsilon2 = 5E-5;
static double offset  =  5.0;  ; 
static double x_0 = 0 + offset; 
static double x_1 = 1 + offset; 
static double x_m = 0.5*(x_0 +x_1); 
static int flag_domain_type = 3;   // 1 for biperiodic, 2 for sector and 3 for circle
static double xi_2 = 0.0;
static double theta      = 30.*M_PI/180.;
static double fric      = 1.00;
static double k_b        = 1;
static double K_subs     = 1;
static double alpha_1    = 5;//3 
static double alpha_2    = 1.; 
static double v_0        = 0.0;

double addGaussianNoise(double mean, double stddev)
{

    std::random_device rd; //Seed for the random number engine
    std::mt19937 generator(rd());   // Mersenne Twister generator
    std::normal_distribution<double> distribution(mean,stddev);

    return distribution(generator); 

}

double  slip(double f,double v0)
{
    double f_offset_1 = 0.1;
    double f_offset_2 = 12;   
    return exp(-alpha_1*(f-f_offset_1)) +  exp(alpha_2*(f-f_offset_2));
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
    SubFillStructure& subFill   = fillStr["dhand"];


    int DOF   = subFill.numDOFs;
    int auxF  = subFill.numAuxF;
    int eNN   = subFill.eNN;
    int nDim  = subFill.nDim;
    int pDim  = subFill.pDim;

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
    
    //[1.3] Global constraints




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
            gq1[d]   += aux_nodes[auxF*i+6] * gNa[pDim*i+d];   
            gq2[d]   += aux_nodes[auxF*i+7] * gNa[pDim*i+d];   
        }
        // Hessians
        for (int d = 0; d < pDim; d++)
            lapphi += ue_nodes[i] * hNa[pDim*pDim*i+pDim*d+d];
    }
    // v[0]  = 1 ; 
    // v[1]  = 0 ;
    double v_mag = sqrt(v[0]*v[0] + v[1]*v[1]);
    double u_mag = sqrt(u[0]*u[0] + u[1]*u[1]);
    double us_mag = sqrt(u_s[0]*u_s[0] + u_s[1]*u_s[1]);
    double f_mag = abs(k_b*(u_mag-us_mag));
    double A = slip(f_mag,1);
    double xi_2_ = xi_2/A;
    double chi_ = chi/A;

    if (A<1)
    {
        chi_ = chi; 
        xi_2_ = xi_2;
    }

    if (A>10)
    {
        A=100;
        chi_ = 0;
        xi_2_ = 0 ;
    } 
    v[0] =  k_b*(u_mag-us_mag);
    v[1] =  0;
    v_mag = sqrt(v[0]*v[0] + v[1]*v[1]);
    double r    = sqrt(x[0]*x[0] +  x[1]*x[1]);
    double kon_ = kon_phi;
    double koff_= koff_phi*A; 
    double mobility_ = mobility;
    double epsilon2_  = epsilon2;


    double advec_param = 1E-5 ; //3E-5;
    double phi_0 = x_0; 
    double phi_1 = x_1;
    double c = 1; 
    double K_stiff =  c*fric;
    double phi_threshold = 9; 
    double threshold_mobility =  1.0 - 0.5*(1+tanh(20*(phin-phi_threshold)));
    mobility_=mobility_*threshold_mobility;
    tensor<double,1> f={0,0};

    double directional_gradient =  v[0]*gphin[0] +  v[1]*gphin[1]; 

    K_stiff  += 1E5*(1 + tanh(100*(phin-6)));


    if (directional_gradient<0)
    {
        f(0) = advec_param*directional_gradient*v[1]/v_mag; 
        f(1) = advec_param*directional_gradient*v[0]/v_mag;
    }
    else if (directional_gradient>0)
    {
	    f(0) = -advec_param*directional_gradient*v[1]/v_mag;
        f(1) = -advec_param*directional_gradient*v[0]/v_mag;
    }


    // Double well and derivatives
    double W_sub   = (pow((phi-phi_0),4)  +  pow(phi_1-phi,4)-1)/12.0;
    double dW_sub = (4*pow((phi-phi_0),3)  -  4*pow(phi_1-phi,3))/12.0;
    double ddW_sub = (12*pow((phi-phi_0),2)  +  12*pow(phi_1-phi,2))/12.0;
    double dddW_sub= (24*(phi-phi_0)  -  24*(phi_1-phi))/12.0; 

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
            fillStr.Ak(0, 0)[i*DOF*eNN+j*DOF] += jac * mobility_ * (ddW + phi * dddW) * Na[j] * gNagphi;
            fillStr.Ak(0, 0)[i*DOF*eNN+j*DOF] += jac * mobility_ * phi * ddW * gNagNb;
            fillStr.Ak(0, 0)[i*DOF*eNN+j*DOF] += jac * epsilon2_ * ((phi * lapNa[i] + gNagphi) * lapNa[j] + (lapNa[i] * Na[j] + gNagNb) * lapphi);

        }
        // Fill RHS
        fillStr.Bk(0)[i*DOF] += jac * ( Na[i] * ((phi-phin) / deltat_   - kon_*(x_m-phin) + koff_ * phin ) + mobility_ * (phi * ddW * gNagphi) + epsilon2_ * (phi * lapNa[i] + gNagphi) * lapphi     );

    }


    wrapper<double,2> Bk(fillStr.Bk(0).data(),eNN,DOF);
    wrapper<double,4> Ak(fillStr.Ak(0,0).data(),eNN,DOF,eNN,DOF);
    tensor<double,2>  id2d = Identity(2);
    wrapper<double,2> nborCoords(subFill.nborCoords.data(),eNN,nDim);
    wrapper<double,2> nborAuxF(subFill.nborAuxF.data(),eNN,DOF);
    wrapper<double,1> bf_(subFill.getDer(0),eNN);
    wrapper<double,2> Dbf_l(subFill.getDer(1),eNN,pDim);


    tensor<double,2> T = nborCoords(all,range(0,1)).T() * Dbf_l;
    //Global derivatives (wrt x and y ) of the basis functions
    tensor<double,2>  Dbf_g = Dbf_l*T.inv();
    double f_mag2 = f*f;
    tensor<double,2> P ={{0,0},{0,0}};
    if (f_mag2>1E-6)
    	P = outer(f,f)/f_mag2;

    tensor<double,1> divP={0,0}; 
    // tensor<double ,1 > u_  = {u[0], u[1]};
    tensor<double,1> gphi_(2); 
    gphi_(0)  = gphi[0]; 
    gphi_(1)  = gphi[1];
    wrapper<double,2> nborDOFs_(subFill.nborDOFs.data(),eNN,DOF);
    wrapper<double,2> nborDOFs0_(subFill.nborDOFs0.data(),eNN,DOF);

    tensor<double,2> M    = xi_2_*P(i,k)  -    xi_2_*xi_2_*P(i,m)*P(m,k)  +    xi_2_*xi_2_*xi_2_*P(i,m)*P(m,n)*P(n,k);

    Bk(all ,0)(K)                  +=  jac*xi_2_*M(i,k)*(phi * ddW*Dbf_g(K,i)*gphi_(k)); 
    Ak(all ,0,all,0)(K,L)          +=  jac*xi_2_*M(i,k)*(phi * ddW*Dbf_g(K,i)*Dbf_g(L,k)  + phi * dddW*bf_(L)*Dbf_g(K,i)*gphi_(k) + bf_(L)*ddW*Dbf_g(K,i)*gphi_(k)); 

    Bk(all ,0)(K)                  +=  jac* epsilon2_*( M(i,k)*gphi_(i)*Dbf_g(K,k)     ) ;
    Ak(all ,0,all,0)(K,L)          +=  jac* epsilon2_*( M(i,k)*Dbf_g(L,i)*Dbf_g(K,k)   ) ; 
 
}



// void LS_(hiperlife::FillStructure& fillStr)
// {
//     using namespace hiperlife;
//     using namespace hiperlife;
//     using namespace std;
//     using Teuchos::RCP;
//     using namespace hiperlife::Tensor;
//     using ttl::index::k, ttl::index::l, ttl::index::m, ttl::index::n, ttl::index::o, ttl::index::p, ttl::index::i; 
//     using ttl::index::K, ttl::index::L;
//     // Rename variables
//     SubFillStructure& subFill = fillStr["dhand"];
//     int DOF   = subFill.numDOFs;
//     int eNN   = subFill.eNN;
//     int nDim  = subFill.nDim;
//     int pDim  = subFill.pDim;
//     int auxF  = subFill.numAuxF;
//     double *xe_nodes  = subFill.nborCoords.data();
//     double *ue_nodes  = subFill.nborDOFs.data();
//     double deltat_       =   fillStr.paramStr->dparam[0];   
//     // Shape functions and derivatives at Gauss points
//     double jac;
//     double* Na = subFill.nborBFs();
//     tensor<double,1> lapNa_(eNN);
//     tensor<double,2> gNa_(eNN,pDim);
//     tensor<double,3> hNa_(eNN,pDim,pDim);

//     std::vector<double> gNa(eNN*pDim), hNa(eNN*pDim*pDim), lapNa(eNN);
//     GlobalBasisFunctions::hessians(gNa.data(), jac, hNa.data(), lapNa.data(), subFill);
//     GlobalBasisFunctions::hessians(gNa_.data(), jac, hNa_.data(), lapNa_.data(), subFill);

//     // Values at integration points
//     std::vector<double> x(pDim), gphi(pDim), gphin(pDim);
//     double phi{}, phin{}, lapphi{},phi2{};
//     double v[2]{},u[2]{},u_s[2]{},nn[2]{},q[2]{},gq1[2]{}, gq2[2]{} ;
//     Array::Fill(x.data(), pDim, 0.0);
//     Array::Fill(gphi.data(), pDim, 0.0);
//     Array::Fill(gphin.data(), pDim, 0.0);
//     double *ue_nodes0 = subFill.nborDOFs0.data();
//     double *aux_nodes = subFill.nborAuxF.data();
    
//     for (int i = 0; i < eNN; i++)
//     {
//         // Values
//         for (int d = 0; d < pDim; d++)
//             x[d] += xe_nodes[nDim*i+d] * Na[i];
//         phi    += ue_nodes[DOF*i+0]  * Na[i];
//         phin   += ue_nodes0[DOF*i+0] * Na[i];
//         v[0]   += aux_nodes[auxF*i+0]  * Na[i];
//         v[1]   += aux_nodes[auxF*i+1] * Na[i];
//         u[0]   += aux_nodes[auxF*i+2]  * Na[i];
//         u[1]   += aux_nodes[auxF*i+3] * Na[i];
//         u_s[0] += aux_nodes[auxF*i+4]  * Na[i];
//         u_s[1] += aux_nodes[auxF*i+5] * Na[i];
        
//         // Gradients
//         for (int d = 0; d < pDim; d++)
//         {
//             gphi[d]  += ue_nodes[DOF*i+0] * gNa[pDim*i+d];
//             gphin[d] += ue_nodes0[DOF*i+0] * gNa[pDim*i+d];   
//         }
//         // Hessians
//         for (int d = 0; d < pDim; d++)
//             lapphi += ue_nodes[i] * hNa[pDim*pDim*i+pDim*d+d];
//     }
//     double v_mag = sqrt(v[0]*v[0] + v[1]*v[1]);
//     double u_mag = sqrt(u[0]*u[0] + u[1]*u[1]);
//     double us_mag = sqrt(u_s[0]*u_s[0] + u_s[1]*u_s[1]);
//     double f_mag = abs(k_b*(u_mag-us_mag));
//     double chi_ = chi; 
//     double A = slip(f_mag,1);
//     if (A>10)
//     {
//         A=1000;
//         chi_ = -chi_;
//     }
//     double r    = sqrt(x[0]*x[0] +  x[1]*x[1]);
//     double kon_ = kon_phi;
//     double koff_= koff_phi*A; 
//     double mobility_ = mobility;
//     double epsilon2_  = epsilon2;
//     double D = 0.0; 
//     double kp_u = 0 ;
//     double kd_u = 0;
//     double advec_param = 1E-5 ; 
//     double phi_0 = x_0; 
//     double phi_1 = x_1;
//     double c = 1; 
//     double K_stiff =  c*fric;
//     double phi_threshold = 9; 
//     double threshold_mobility =  1.0 - 0.5*(1+tanh(20*(phin-phi_threshold)));
//     mobility_=mobility_*threshold_mobility;
//     tensor<double,1> f={0,0};
//     tensor<double,1> nematic_force = {0, 0};
//     double S  = 2*pow(q[0]*q[0] + q[1]*q[1] ,0.5);
//     tensor<double,1> div_S(2); 
//     div_S(0) = 2*pow(q[0]*q[0] + q[1]*q[1]  ,-0.5)*(gq1[0]*q[0] + gq2[0]*q[1]);
//     div_S(1) = 2*pow(q[0]*q[0] + q[1]*q[1]  ,-0.5)*(gq1[1]*q[0] + gq2[1]*q[1]);

//     // Calculate the factor: -0.5 * (1 - tanh^2(-phi + 6))
//     double tanhTerm = tanh(6.0-phin);
//     double factor = -0.5 * (1 - tanhTerm * tanhTerm);
//     tensor<double,1> gradient_lambda(2);
//     gradient_lambda(0)  = factor*gphin[0];
//     gradient_lambda(1)  = factor*gphin[1];

//     if (S>1E-6 ) 
//     {
//        nematic_force(0)  = gq1[0] + gq2[1] + 0.5*div_S(0) + gradient_lambda(0)*(q[0] + 0.5*S)  + gradient_lambda(1)*q[1]   ; //pow(q[0]/S + 0.5 ,0.5) ; 
//        nematic_force(1)  = gq2[0] - gq1[1] + 0.5*div_S(1) + gradient_lambda(0)*q[1] +gradient_lambda(1)*(-q[0] + 0.5*S)    ; //  q[1]/(S*nematic_force(0));
//     }
 
//     double nematic_force_mag = sqrt(nematic_force(0)*nematic_force(0)   +   nematic_force(1)*nematic_force(1)  );

//     double directional_gradient =  v[0]*gphin[0] +  v[1]*gphin[1]; 
//     double xi_2_ = xi_2; 
//     K_stiff  += 1E5*(1 + tanh(100*(phin-6)));

//     // Double well and derivatives
//     double W_sub   = (pow((phi-phi_0),4)  +  pow(phi_1-phi,4)-1)/12.0;
//     double dW_sub = (4*pow((phi-phi_0),3)  -  4*pow(phi_1-phi,3))/12.0;
//     double ddW_sub = (12*pow((phi-phi_0),2)  +  12*pow(phi_1-phi,2))/12.0;
//     double dddW_sub= (24*(phi-phi_0)  -  24*(phi_1-phi))/12.0; 

//     double W = W_sub + 0.5 * chi_ * (phi-phi_0) * (phi-phi_0) ;
//     double dW = dW_sub + chi_ * (phi-phi_0)    ;
//     double ddW = ddW_sub  + chi_  ;
//     double dddW = dddW_sub  ;

//     // Fill
//     for (int i = 0; i < eNN; i++)
//     {
//         // Gradient of basis function times gradient of phase-field variableñ
//         double gNagphi{};
//         for (int d = 0; d < pDim; d++)
//             gNagphi += gNa[pDim*i+d]*gphi[d];

//         for (int j = 0; j < eNN; j++)
//         {
//             // Product of the gradient of basis functions
//             double gNagNb{};
//             double  gNb{};
//             for (int d = 0; d < pDim; d++)
//             {
//                 gNagNb += gNa[pDim*i+d]*gNa[pDim*j+d];
//             }
//             // Fill Ak
//             fillStr.Ak(0, 0)[i*DOF*eNN+j*DOF] += jac * Na[i] * (Na[j] * (1./deltat_)   );
//             fillStr.Ak(0, 0)[i*DOF*eNN+j*DOF] += jac * mobility_ * (ddW + phi * dddW) * Na[j] * gNagphi;
//             fillStr.Ak(0, 0)[i*DOF*eNN+j*DOF] += jac * mobility_ * phi * ddW * gNagNb;
//             fillStr.Ak(0, 0)[i*DOF*eNN+j*DOF] += jac * epsilon2_ * ((phi * lapNa[i] + gNagphi) * lapNa[j] + (lapNa[i] * Na[j] + gNagNb) * lapphi);
//             // Fill RHS
//        }
//        fillStr.Bk(0)[i*DOF] += jac * ( Na[i] * ((phi-phin) / deltat_   - kon_*(x_m-phin)-kp_u+ kd_u*phin + koff_ * phin ) + mobility_ * (phi * ddW * gNagphi) + epsilon2_ * (phi * lapNa[i] + gNagphi) * lapphi     );
 
//     }
    
// }


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

    double gamma  = 0.01;
    double visc   = 0.01;
    double fric_ = fric;
    double tau  = 1;
    double tau_2 = 1;
    // double gamma  = 0.0109/5.5;
    // double visc   = 0.001;
    tensor<double,1> v_star = {v_0,0};
    double deltat_       =   1E4*fillStr.paramStr->dparam[0];   

    SubFillStructure& g_subFill = fillStr["gloDHand"];
    int g_numDOFs = g_subFill.numDOFs;
    wrapper<double,1> gDOFs(g_subFill.nborDOFs.data(),g_numDOFs);
    double V = gDOFs(0);
 

    // Rename variables for the main dofsHandler
    wrapper<double,2> nborCoords(subFill.nborCoords.data(), eNN, nDim);
    wrapper<double,2> nborDOFs(subFill.nborDOFs.data(), eNN, DOF);
    wrapper<double,2> nborDOFs0(subFill.nborDOFs0.data(), eNN, DOF);
    wrapper<double,2> nborauxF(subFill.nborAuxF.data(), eNN, auxF);
    wrapper<double,1> bf(subFill.nborBFs(), eNN);
    wrapper<double,2> Dbf_l(subFill.getDer(1),eNN,pDim);
    tensor<double,2>  id2d = Identity(2);

    // Get output
    wrapper<double,2> Bk(fillStr.Bk(0).data(),eNN,DOF);
    wrapper<double,1> Bg(fillStr.Bk(1).data(),g_numDOFs);
    wrapper<double,4> Ak(fillStr.Ak(0,0).data(),eNN,DOF,eNN,DOF);
    wrapper<double,3> Akg(fillStr.Ak(0,1).data(),eNN,DOF,g_numDOFs);
    wrapper<double,3> Agk(fillStr.Ak(1,0).data(),g_numDOFs,eNN,DOF);
    wrapper<double,2> Agg(fillStr.Ak(1,1).data(),g_numDOFs,g_numDOFs);


    tensor<double,2> T       =  (nborCoords(all,range(0,1))(L,k)*Dbf_l(L,l))(k,l);
    tensor<double,2> Tinv    =  T.inv();   

    tensor<double,2> Dbf_g      =  Dbf_l(K,k)*Tinv(k,l);
    tensor<double,2> nbor_v     =  nborDOFs(all,range(0,1));
    tensor<double,2> nbor0_v     =  nborDOFs0(all,range(0,1));
    tensor<double,1> nbor_phi    =  nborauxF(all,0);
    tensor<double,1>  u  =  bf(K)* nborDOFs(all,range(2,3))(K,k);
    tensor<double,1>  x  =  bf(K)* nborCoords(all,range(0,2))(K,k);
    tensor<double,1>  u0 =  bf(K)* nborDOFs0(all,range(2,3))(K,k);
    double u0_mag = sqrt(u0(k)*u0(k));
    tensor<double,1> v       =  bf(K) * nbor_v(K,k);
    tensor<double,1> v0      =  bf(K) * nbor0_v(K,k);

    double v0_mag = sqrt(v0(k)*v0(k));


    tensor<double,2> nbor_us      =  nborDOFs(all,range(4,5)); 
    tensor<double,2> nbor0_us     =  nborDOFs0(all,range(4,5));  
    
    tensor<double,1>  us          =  bf(K)* nbor_us(K,k);
    tensor<double,1>  us_0        =  bf(K)* nbor0_us(K,k);
    tensor<double,1>  v_s          = (us-us_0)/deltat_;
    tensor<double,2> Du_s  =  (nbor_us(K,m)*Dbf_g(K,n))(m,n);    
    double us0_mag = sqrt(us_0(k)*us_0(k));

    double phi     =  bf(K) * nbor_phi(K);

    tensor<double,1> nbor0_c     =  nborDOFs0(all,6);
    tensor<double,1> nbor_c      =  nborDOFs(all,6);
    double c0     =  bf(K) * nbor0_c(K);
    double c      =  bf(K) * nbor_c(K);    

    tensor<double,2> Dv          =  (nbor_v(K,m)*Dbf_g(K,n))(m,n);
    double jac = T.det();

    tensor<double,4> dvDv = (Dbf_g(K,n)*id2d(m,k))(K,k,m,n) ; 
    tensor<double,2> d_divv  =  (dvDv(K,k,l,m)*id2d(l,m))(K,k);


    tensor<double,2> Idd(2,2) ;
    Idd(0,0)                  = 1;
    Idd(1,0)                  = 0; 
    Idd(0,1)                  = 0;    
    Idd(1,1)                  = 0; 
    tensor<double,2> rodt     = (0.5*(Dv(m,n)  + Dv(n,m)))(m,n) + V*Idd(m,n); 

    double divv = rodt(0,0) + rodt(1,1)  + V  ;
    tensor<double,2> d_divu  =  (dvDv(K,k,l,m)*id2d(l,m))(K,k);
    tensor<double,4> dv_rodt  = (0.5*(dvDv(K,k,m,n)  + dvDv(K,k,n,m)))(K,k,m,n);
    tensor<double,2> dV_rodt(2,2);  
    dV_rodt   = Idd;

    //#######################################################################################//   
    //###############################  ACTIN CORTEX EQUATIONS ##############################//
    //#####################################################################################//

    // //[9.2] GRADIENT
    Bk(all,range(0,1))(K,k)                    +=  jac * gamma *  d_divv(K,k);
    Bk(all,range(0,1))(K,k)                    +=  jac * visc  * (dv_rodt(K,k,m,n)*rodt(m,n))(K,k);
    // //   //[12.2] HESSIAN
    Ak(all,range(0,1),all,range(0,1))(K,k,L,l) +=  jac * visc * dv_rodt(K,k,m,n)*dv_rodt(L,l,m,n);


    Bg(0)                                      +=  jac*gamma;
    Bg(0)                                      +=  jac * visc * dV_rodt(m,n)*rodt(m,n);
    Agg(0,0)                                   +=  jac * visc * dV_rodt(m,n)*dV_rodt(m,n);
    Agk(0,all,range(0,1))(K,k)                 +=  jac * visc * dV_rodt(m,n)*dv_rodt(K,k,m,n);
    Akg(all,range(0,1),0)(K,k)                 +=  jac * visc * dV_rodt(m,n)*dv_rodt(K,k,m,n);
    //#######################################################################################//   
    //###############################  FRICTION FORCE FROM ENV #############################//
    //#####################################################################################//

    tensor<double,1> Id_v(2);
    Id_v(0) = 1;     Id_v(1) = 0; 
    Bk(all,range(0,1))(K,k)                            +=    fric_*jac* bf(K)*(v(k)  + V*x(0)*Id_v(k));
    Ak(all ,range(0,1),all ,range(0,1))(K,k,L,l)       +=    fric_*jac* bf(K)*bf(L)*id2d(k,l);
    Akg(all,range(0,1),0)(K,k)                         +=    fric_*jac* bf(K)*x(0)*Id_v(k);   

    Bg(0)                                              +=    fric_*jac*(v(k) + (V*x(0)*Id_v)(k)  )*Id_v(k)*x(0);
    Agk(0,all,range(0,1))(K,k)                         +=    fric_*jac*Id_v(k)*x(0)*bf(K);
    Agg(0,0)                                           +=    fric_*jac*Id_v(k)*x(0)*Id_v(k)*x(0);
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
    int nEl = 5;

    // Time-related parameters
    string oname   = "integrin.";
    string oname_v = "velocity.";
    double maxTime  = 10000000;
    double maxNStep = 10000000;
    double maxDelt  = 1E-4;
    double stepFactor = 0.9;
    int nSave = 1;
    double width = 0.1; 
    double L = 1.0;

    SmartPtr<ParamStructure> userStr   = Create<ParamStructure>() ;
    userStr->dparam.resize(50);
    userStr->dparam[0] = deltat;
    // Generate mesh
    SmartPtr<StructMeshGenerator> structMesh = Create<StructMeshGenerator>();
    try
    {
        structMesh->setMesh(eType, bfType, 1);
        structMesh->setPeriodicBoundaryCondition({Axis::Yaxis});
        structMesh->genRectangle(nEl, 2, L, L*0.01);
    }
    catch (runtime_error& err)
    {
        throw runtime_error("Mesh could not be generated.");
    }

    Teuchos::RCP<MeshLoader> loadedMesh = Teuchos::rcp(new MeshLoader);
    loadedMesh->setElemType(ElemType::Triang);
    loadedMesh->setBasisFuncType(BasisFuncType::SubdivSurfs);
    loadedMesh->setBasisFuncOrder(bfOrder);
    loadedMesh->loadMesh("/home/mirza/src/apps/endothelium_modelling/circle_n4.vtk",hiperlife::MeshType::Parallel);  

    // Create distributed mesh
    SmartPtr<DistributedMesh> disMesh = Create<DistributedMesh>();
    try
    {
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
  

    SmartPtr<DistributedMesh> gloDisMesh = Create<DistributedMesh>();

    try
    {

        gloDisMesh->setMeshRelation(MeshRelation::GlobConstr, disMesh);
        gloDisMesh->setBalanceMesh(false);
        gloDisMesh->Update();          
    }
    catch (runtime_error& err)
    {
        cout << myRank << ": gloDHand could not be created " << err.what() << endl;
        MPI_Finalize();
        return 1;
    
    }
    SmartPtr<DOFsHandler> gloDHand = Create<DOFsHandler>(gloDisMesh);
    try
    {
        gloDHand->setNameTag("gloDHand");
        gloDHand->setDOFs({"V"});
        gloDHand->Update();
    }
    catch (runtime_error& err)
    {
        cout << myRank << ": gloDHand could not be created " << err.what() << endl;
        MPI_Finalize();
        return 1;
    }


    try
    {
        dhand_v->setNameTag("dhand_v");
        dhand_v->setDOFs({"vx","vy","ux","uy","us_x","us_y","c"});
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
        double mean  = kon_phi * x_m / (kon_phi + koff_phi) ; 
        // Calculate the value of r using the sine wave
        double r  = AMPLITUDE * std::sin(phase) ;
        double r1 = 0.001 * (((double) rand() / (RAND_MAX)) * 2.0 - 1.0);
        double r2 = addGaussianNoise(mean, 0.01*mean);
        // Set the initial value with the sine wave perturbation

        dhand->nodeDOFs->setValue(0, i, IndexType::Local,   r2);
        int crease = dhand_v->mesh->nodeCrease(i, IndexType::Local);
        if (x<1E-2)
        {
            dhand_v->nodeDOFs->setValue("us_x", i, IndexType::Local, 0);
            //dhand_v->setConstraint("us_x", i, IndexType::Local, 0); 
        }


    }
    dhand_v->nodeDOFs->setValue("c",kon/(koff*slip(0,v_0))); 
    dhand->nodeDOFs0->setValue(dhand->nodeDOFs);
    dhand_v->nodeDOFs0->setValue(dhand_v->nodeDOFs);
    dhand_v->nodeDOFs->setValue("vx",v_0);
    dhand_v->nodeDOFs->setValue("vy",0);
    dhand_v->nodeDOFs->setValue("ux",0);
    dhand_v->nodeDOFs->setValue("uy",0);  
    dhand_v->nodeDOFs->setValue("us_x",0);   
    dhand_v->nodeDOFs->setValue("us_y",0);

    //gloDHand->setConstraint("V",0);
    dhand_v->setConstraint("vy",0);      
    dhand_v->setConstraint("uy",0); 
    dhand_v->setConstraint("c",0); 
    dhand_v->setConstraint("us_x",0); 
    dhand_v->setConstraint("us_y",0);   
    dhand_v->setConstraint("ux",0); 
    dhand_v->setConstraint("uy",0);  
    dhand->UpdateGhosts();
    gloDHand->UpdateGhosts();

    // Create HiPerProblem
    SmartPtr<HiPerProblem> hiperProbl = Create<HiPerProblem>();
    try
    {
        // Set DOFHandler
        hiperProbl->setDOFsHandlers({dhand});

        // Set Integration
        hiperProbl->setIntegration("Integ", {"dhand"});
        hiperProbl->setCubatureGauss("Integ", gPts);
        hiperProbl->setElementFillings("Integ", LS);
        hiperProbl->setParameterStructure(userStr);
        hiperProbl->setIntegration("BorderInteg", {"dofHand"});
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
        hiperProbl_v->setDOFsHandlers({dhand_v,gloDHand});
        // Set Integration
        hiperProbl_v->setIntegration("Integ", {"dhand","gloDHand"});
        hiperProbl_v->setCubatureGauss("Integ",4);
        hiperProbl_v->setParameterStructure(userStr);
        hiperProbl_v->setElementFillings("Integ", ConsistencyCheck<LS_v>);
        //hiperProbl_v->setElementFillings("Integ", LS_v);
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
    linSolver_v->setWorkSpaceMemoryIncrease(200);
    linSolver_v->setDefaultParameters();
    linSolver_v->Update();


    // Create nonlinear solver
    SmartPtr<NewtonRaphsonNonlinearSolver> nonlinSolver = Create<NewtonRaphsonNonlinearSolver>();
    nonlinSolver->setLinearSolver(linSolver);
    nonlinSolver->setMaxNumIterations(20);
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
    nonlinSolver_v->setMaxNumIterations(20);
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
    outFile << "time" << "," << "vx" << "," << "c" <<  "," << "ux"<< std::endl; 
    while ((timeStep < maxNStep) and (time < maxTime))
    {
        // Write
        if (dhand->myRank() == 0)
            cout << endl << " TS: " << to_string(timeStep) << " Time " << time <<  " Delta " << deltat << endl;

        // Initial guess
        dhand->nodeDOFs->setValue(dhand->nodeDOFs0);
        hiperProbl->UpdateGhosts();

        // Newton-Raphson method
        int    ierr  = true; //nonlinSolver->solve();

        double max = -10;
        double min = 10;
        for(int i = 0; i < disMesh->loc_nPts(); i++)
        {
            double val = dhand->nodeDOFs->getValue(0,i,IndexType::Local);
            max = std::max(max,val);
            min = std::min(min,val);
        }
        MPI_Allreduce(MPI_IN_PLACE,&max,1,MPI_DOUBLE,MPI_MAX,dhand->comm());
        MPI_Allreduce(MPI_IN_PLACE,&min,1,MPI_DOUBLE,MPI_MIN,dhand->comm());
        
        // cout << dhand->myRank() << " " << min << " " << max << endl;
        // Check convergence
        if (ierr != 0)// and max < 1.0 and min > 0.0)
        {
            // Save solution
            dhand->nodeDOFs0->setValue(dhand->nodeDOFs);
            // Initial guess
            dhand_v->nodeDOFs->setValue(dhand_v->nodeDOFs0);
            gloDHand->nodeDOFs->setValue(gloDHand->nodeDOFs0);
            hiperProbl->UpdateGhosts();
            // Newton-Raphson method
            int ierr_v = nonlinSolver_v->solve();

            if (ierr_v != 0)
            {


                for (int i = 0; i < dhand->mesh->loc_nPts(); i++)
                {

                    double vx  = dhand_v->nodeDOFs->getValue("vx", i, IndexType::Local);
                    double vy  = dhand_v->nodeDOFs->getValue("vy", i, IndexType::Local);
                    double ux  = dhand_v->nodeDOFs->getValue("ux", i, IndexType::Local);
                    double uy  = dhand_v->nodeDOFs->getValue("uy", i, IndexType::Local);
                    double us_x  = dhand_v->nodeDOFs->getValue("us_x", i, IndexType::Local);
                    double us_y  = dhand_v->nodeDOFs->getValue("us_y", i, IndexType::Local);

                    dhand->nodeAuxF->setValue("vx", i, IndexType::Local, vx);
                    dhand->nodeAuxF->setValue("vy", i, IndexType::Local, vy);
                    dhand->nodeAuxF->setValue("ux", i, IndexType::Local, ux);
                    dhand->nodeAuxF->setValue("uy", i, IndexType::Local, uy);
                    dhand->nodeAuxF->setValue("us_x", i, IndexType::Local, us_x);
                    dhand->nodeAuxF->setValue("us_y", i, IndexType::Local, us_y);


                }
                dhand->UpdateGhosts();
                // Save solution
                dhand_v->nodeDOFs0->setValue(dhand_v->nodeDOFs);
                gloDHand->nodeDOFs0->setValue(gloDHand->nodeDOFs);
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
                    double c   = dhand_v->nodeDOFs->getValue("c",  0, IndexType::Local);

                    outFile << time << "," << vx << "," << c <<  "," << ux<< std::endl; 
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
                gloDHand->nodeDOFs->setValue(gloDHand->nodeDOFs0);
            }
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
