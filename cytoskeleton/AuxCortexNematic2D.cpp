#include <iostream>
#include <mpi.h>
#include <fstream>

/// Trilinos headers
#include <Teuchos_RCP.hpp>

/// hiperlife headers
#include "hl_TypeDefs.h"
#include "hl_GlobalBasisFunctions.h"
#include "hl_StructMeshGenerator.h"
#include "hl_DistributedMesh.h"
#include "hl_FillStructure.h"
#include "hl_DOFsHandler.h"
#include "hl_HiPerProblem.h"
#include "hl_LinearSolver_Direct_Amesos.h"
#include "hl_Tensor.h"
#include "hl_LinearSolver_Iterative_AztecOO.h"
#include "AuxCortexNematic2D.h"
#include "cmath"
#include <random>
#include "hl_SurfLagrParam.h"

double  catch_slip_rate(double f)
{
    return exp(-3*f);
}

void LS_phi(hiperlife::FillStructure& fillStr)
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

    double deltat_    = 1E-2*fillStr.paramStr->dparam[0];
    double kon_phi  =   fillStr.paramStr->dparam[26];
    double koff_phi =   fillStr.paramStr->dparam[27];
    double x_m      =   fillStr.paramStr->dparam[28];
    double x_0      =   fillStr.paramStr->dparam[29];
    double x_1      =   fillStr.paramStr->dparam[30];
    double chi      =   fillStr.paramStr->dparam[31];
    double epsilon2 =   fillStr.paramStr->dparam[32];  
    double mobility =   fillStr.paramStr->dparam[33]; 

    // cout<<kon_phi<<" "<<koff_phi<<" "<<x_m<<" "<<x_0<<" "<<x_1<<" "<<chi<<" "<<epsilon2<<" "<<" "<<mobility<<endl;

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
    double phi{}, phin{}, lapphi{};
    double v[2]{},u[2]{} ;
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
        phi    += ue_nodes[DOF*i]  * Na[i];
        phin   += ue_nodes0[DOF*i] * Na[i];
        v[0]   += aux_nodes[auxF*i]  * Na[i];
        v[1]   += aux_nodes[auxF*i+1] * Na[i];
        u[0]   += aux_nodes[auxF*i+2]  * Na[i];
        u[1]   += aux_nodes[auxF*i+3] * Na[i];

        // Gradients
        for (int d = 0; d < pDim; d++)
        {
            gphi[d]  += ue_nodes[DOF*i] * gNa[pDim*i+d];
            gphin[d] += ue_nodes0[DOF*i] * gNa[pDim*i+d];    
        }
        // Hessians
        for (int d = 0; d < pDim; d++)
            lapphi += ue_nodes[DOF*i] * hNa[pDim*pDim*i+pDim*d+d];
    }

    double v_mag = sqrt(v[0]*v[0] + v[1]*v[1]);
    double u_mag = sqrt(u[0]*u[0] + u[1]*u[1]);
    double f_mag = abs(u_mag);
    double A = catch_slip_rate(f_mag);
    double A_catch = catch_slip_rate(f_mag);
    double chi_ = chi/A_catch;
    double r    = sqrt(x[0]*x[0] +  x[1]*x[1]);
    if (r<4.0)
    {
        chi_ = 0.0 ; 
    }
    double kon_ = kon_phi;
    double koff_= koff_phi*A; 
    double mobility_ = mobility;
    double epsilon2_  = epsilon2;
    double advec_param = 1E-5 ; //3E-5;
    double phi_0 = x_0; 
    double phi_1 = x_1;
    double phi_threshold = 9;
    double threshold_mobility =  1.0 - 0.5*(1+tanh(20*(phin-phi_threshold)));

    double alpha_log =0;
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
            fillStr.Ak(0, 0)[i*DOF*eNN+j*DOF] += jac * mobility_ * (ddW + phi * dddW) * Na[j] * gNagphi;
            fillStr.Ak(0, 0)[i*DOF*eNN+j*DOF] += jac * mobility_ * phi * ddW * gNagNb;
            fillStr.Ak(0, 0)[i*DOF*eNN+j*DOF] += jac * epsilon2_ * ((phi * lapNa[i] + gNagphi) * lapNa[j] + (lapNa[i] * Na[j] + gNagNb) * lapphi);

        }
        // Fill RHS
        fillStr.Bk(0)[i*DOF] += jac * ( Na[i] * ((phi-phin) / deltat_   - kon_*(x_m-phin) + koff_ * phin ) + mobility_ * (phi * ddW * gNagphi) + epsilon2_ * (phi * lapNa[i] + gNagphi) * lapphi     );

    }


 
}


void LS_border_phi(hiperlife::FillStructure& fillStr)
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
            lapphi += ue_nodes[DOF*i] * hNa[pDim*pDim*i+pDim*d+d];
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





double addGaussianNoise(double mean, double stddev)
{

    std::random_device rd; //Seed for the random number engine
    std::mt19937 generator(rd());   // Mersenne Twister generator
    std::normal_distribution<double> distribution(mean,stddev);

    return distribution(generator); 

}

void LS_CortexNematic_2D(hiperlife::FillStructure& fillStr)
{
    using namespace hiperlife;
    using namespace hiperlife::Tensor;
    using ttl::index::k, ttl::index::l, ttl::index::m, ttl::index::n, ttl::index::o, ttl::index::p; 
    using ttl::index::K, ttl::index::L;


    //-----------------------------------------------------------
    //[1] INPUT DATA
    auto& subFill = fillStr["cortexHand"];
    int nDim = subFill.nDim;
    int pDim = subFill.pDim;
    int eNN  = subFill.eNN;
    int numDOFs = subFill.numDOFs;
    int numAuxF = subFill.numAuxF;

    wrapper<double,2> nborCoords(subFill.nborCoords.data(),eNN,nDim);
    wrapper<double,2> nborDOFs(subFill.nborDOFs.data(),eNN,numDOFs);
    wrapper<double,2> nborAuxF(subFill.nborAuxF.data(),eNN,numAuxF);
    wrapper<double,2> nborDOFs0(subFill.nborDOFs0.data(),eNN,numDOFs);

    wrapper<double,1> bf(subFill.getDer(0),eNN);
    wrapper<double,2> Dbf_l(subFill.getDer(1),eNN,pDim);
    wrapper<double,3> DDbf_l(subFill.getDer(2),eNN,pDim,pDim);
    //tensor<double,2> Dbf_g(eNN,pDim);
    tensor<double,3> DDbf_g(eNN,pDim,pDim);
    //tensor<double,1> Lapbf_g(eNN);

    double deltat = fillStr.paramStr->dparam[0];
    double sus20  = fillStr.paramStr->dparam[1];
    double sus40  = fillStr.paramStr->dparam[2];
    double hcrit  = fillStr.paramStr->dparam[3];

    double kp  = fillStr.paramStr->dparam[4];
    double kd0 = fillStr.paramStr->dparam[5];
    double frank  = fillStr.paramStr->dparam[6];


    double lambda_iso    = fillStr.paramStr->dparam[7];
    double lambda_aniso  = fillStr.paramStr->dparam[8];
    double lambda_rot    = fillStr.paramStr->dparam[9];

    double visc    = fillStr.paramStr->dparam[10];
    double rvisc   = fillStr.paramStr->dparam[11];
    double cvisc   = fillStr.paramStr->dparam[12];

    double fric    = fillStr.paramStr->dparam[13];
    double stab    = fillStr.paramStr->dparam[14];
    double kosm    = fillStr.paramStr->dparam[20];
    double kgrad   = fillStr.paramStr->dparam[21];
    double heqb    = fillStr.paramStr->dparam[22];
    double D       = fillStr.paramStr->dparam[23];
    double lambda_trans =  fillStr.paramStr->dparam[24];
    double x_m      =   fillStr.paramStr->dparam[28];

    double kd;
    tensor<double,1> dkp(pDim);
    tensor<double,1> dkd(pDim);
    //kp=0.0;kd=0.0;
    dkp = 0.0;dkd = 0.0;

    // Re-scaling the coefficients with time step
    sus20  = sus20/deltat;
    sus40  = sus40/deltat;
    frank = frank/deltat;
    kosm  = kosm/deltat;
    kgrad = kgrad/deltat;

    //-----------------------------------------------------------
    //[2] OUTPUT
    wrapper<double,2> Bk(fillStr.Bk(0).data(),eNN,numDOFs);
    wrapper<double,4> Ak(fillStr.Ak(0,0).data(),eNN,numDOFs,eNN,numDOFs);




    double rayleighian{};
    double freeEnergyPotential{};
    double dissipation{};
    double power{};

    //-----------------------------------------------------------
    //[3] GEOMETRY
    tensor<double,1> x = bf * nborCoords;
    //Transformation matrix (from reference to spatial)
    tensor<double,2> T = nborCoords(all,range(0,1)).T() * Dbf_l;
    //Jacobian of the transformation (for integration)
    double jac = T.det();



    //cout<<lambda_trans<<endl;
    //Global derivatives (wrt x and y ) of the basis functions
    //    GlobalBasisFunctions::hessians(Dbf_g, jac, DDbf_g, Lapbf_g, eNN, pDim, nDim, nborCoords, Dbf_l, DDbf_l);
    tensor<double,2>  Dbf_g = Dbf_l*T.inv();
    //-----------------------------------------------------------
    //[4] VARIABLES

    //[4.1] AUXILIARY
    tensor<double,1> nbor_phi = nborAuxF(all,0); 
    tensor<double,1> nbor_h   = nborDOFs(all,0); 
    tensor<double,1> nbor_h0  = nborDOFs0(all,0);
    tensor<double,2> nbor_q   = nborDOFs(all,range(1,2));
    tensor<double,2> nbor_q0  = nborDOFs0(all,range(1,2));
    tensor<double,2> nbor_v   = nborDOFs(all,range(3,4));
    tensor<double,2> nbor_u_sub    = nborDOFs(all,range(5,6));
    tensor<double,2> nbor_u0_sub   = nborDOFs0(all,range(5,6));

    tensor<double,2> id2d = {{1.0,0.0},{0.0,1.0}};
    tensor<double,3> voigt = {{{1,0},{0,1}},
        {{0,1},{-1,0}}};
    tensor<double,4> bf_voigt = outer(bf,voigt.transpose({2,0,1}));
    tensor<double,5> Dbf_voigt = outer(Dbf_g,voigt.transpose({2,0,1})).transpose({0,2,3,4,1});

    //[4.2] STATE VARIABLES
    double h              = bf * nbor_h;
    double h0             = bf * nbor_h0;
    double phi            = bf * nbor_phi;
    tensor<double,1> Dh   = nbor_h * Dbf_g;
    tensor<double,1> Dh0  = nbor_h0 * Dbf_g;
    tensor<double,2> DDh  = nbor_h  * DDbf_g;

    tensor<double,2> q  = product(bf_voigt,nbor_q, {{0,0},{1,1}});
    tensor<double,2> q0 = product(bf_voigt,nbor_q0,{{0,0},{1,1}});
    tensor<double,3> Dq = product(nbor_q,Dbf_voigt, {{0,0},{1,1}});
    tensor<double,3> Dq_0 = product(nbor_q0,Dbf_voigt, {{0,0},{1,1}});


    //[4.3] VELOCITY
    tensor<double,1> v  = bf * nbor_v;
    tensor<double,1> u_sub   =  bf * nbor_u_sub;
    tensor<double,1> u0_sub  =  bf * nbor_u0_sub;
    tensor<double,1> v_sub      =  (u_sub -   u0_sub)/deltat;

    //Velocity gradient
    tensor<double,2> Dv = product(nbor_v,Dbf_g,{{0,0}});
    tensor<double,4> dvDv =(Dbf_g(K,n)*id2d(m,k))(K,k,m,n) ; 
    tensor<double,3> DDv = product(nbor_v,DDbf_g,{{0,0}});
    double divv = trace(Dv);
    tensor<double,1> Ddivv(pDim);
    Ddivv(0) = DDv(0,0,0) +  DDv(1,1,0);
    Ddivv(1) = DDv(0,0,1) +  DDv(1,1,1);

    //Rate-of-deformation tensor
    tensor<double,2> rodt = 0.5*(Dv + Dv.transpose({1,0}));
    tensor<double,4> dvrodt = 0.5 * ( outer(Dbf_g,id2d).transpose({0,3,1,2}) + outer(Dbf_g,id2d).transpose({0,3,2,1}) );

    //Vorticity tensor
    tensor<double,2> omega = 0.5*(Dv - Dv.transpose({1,0}));
    tensor<double,4> dvomega = 0.5 * ( outer(Dbf_g,id2d).transpose({0,3,2,1}) - outer(Dbf_g,id2d).transpose({0,3,1,2}) );

    //[4.4] JAUMANN DERIVATIVE
    tensor<double,2> Jq = (q-q0)(m,n)/deltat  + Dq_0(m,n,o) * v(o) -  omega(m,o)*q0(o,n) +  q0(m,o)*omega(o,n);

    tensor<double,2> Jq_ = (q-q0)(m,n)/deltat + Dq(m,n,o) * v(o) -  omega(m,o)*q(o,n) +  q(m,o)*omega(o,n);
    tensor<double,4> dqJq_ = (bf_voigt(K,k,m,n)/deltat  +  Dbf_voigt(K,k,m,n,o) * v(o) -  omega(m,o)*bf_voigt(K,k,o,n) +  bf_voigt(K,k,m,o)*omega(o,n))(K,k,m,n);
    tensor<double,4> dvJq_ =  (Dq(m,n,k) * bf(K) -  dvomega(K,k,m,o)*q(o,n) +  q(m,o)*dvomega(K,k,o,n))(K,k,m,n);


    tensor<double,4> dqJq = bf_voigt/deltat ;
    tensor<double,4> dvJq =  (Dq_0(m,n,k) * bf(K) -  dvomega(K,k,m,o)*q0(o,n) +  q0(m,o)*dvomega(K,k,o,n))(K,k,m,n);

    tensor<double,6> dqdvJq_ = ( Dbf_voigt(L,l,m,n,k) * bf(K) -  dvomega(K,k,m,o)*bf_voigt(L,l,o,n) +  bf_voigt(L,l,m,o)*dvomega(K,k,o,n))(K,k,L,l,m,n); 

    tensor<double,6> dvdqJq_ = dqdvJq_.transpose({2,3,0,1,4,5});

    double S2 = 2.0*product(q,q,{{0,0},{1,1}});
    double S2_0 = 2.0*product(q0,q0,{{0,0},{1,1}});

    if (sqrt(S2_0)-0.5>0)
    {
        //rvisc += 1E4*(sqrt(S2_0)-0.5);
    }
    //kd = kd0*(1-sqrt(S2_0));


    //   if (abs(lambda_rot)>1e-3 and sqrt(S2_0)>0.5)
    //   {
    //            kd = .1; 
    //            kp = .1*h0;
    //   sus40 = 800;
    //   lambda_rot = -1500;
    //   }
    //   else 
    {
        kd = kd0;
        kp = kp;
    }                                             


    //kd += 100*(1+tanh(100*(h0-1.0)));
    //kp += 100*(1+tanh(100*(h0-1.0)));

    // Defining a spatial field for the polymerization and de-polymerization rate
    //compTurnOverSpaFld_mec1(kp,kd,dkp,dkd,heqb,kd0,x);

    tensor<double,3> dv_v  = bf(K) * id2d(m,n) ;
    tensor<double,3> dvsub_v_sub  = bf(K) * id2d(m,n);

    double h_ = h0 + deltat * (- (v * Dh) - h * divv + kp -kd*h);

    tensor<double,1> Dh_ = Dh0 + deltat * (-Dh*Dv -v*DDh - Dh*divv - h*Ddivv + dkp - kd*Dh - h*dkd);

    tensor<double,2> dhDh_      = deltat * (-Dbf_g*Dv - product(v,DDbf_g,{{0,1}}) - divv*Dbf_g - outer(bf,Ddivv) - kd*Dbf_g - outer(bf,dkd));
    tensor<double,4> dhdvDh_    = deltat * (-product(Dbf_g,dvDv,{{1,2}}) - outer(bf,DDbf_g).transpose({1,0,2,3}) - outer(Dbf_g,Dbf_g).transpose({0,2,3,1}) - outer(bf,DDbf_g) );

    tensor<double,3> dvDh_      = deltat * (-product(Dh,dvDv,{{0,2}}) - outer(bf,DDh) - outer(Dbf_g,Dh) - h*DDbf_g);
    tensor<double,4> dvdhDh_    = dhdvDh_.transpose({1,2,0,3});

    //[5] FREE ENERGY (SUSCEPTIBILITY)
    double sus2{}, dsus2{}, ddsus2{};
    //computeSus2(sus2, dsus2, ddsus2, h_, sus20, hcrit);
    // double ccrit = hcrit ;
    computeSus2(sus2, dsus2, ddsus2, h0, sus20, hcrit);

    double sus4{}, dsus4{}, ddsus4{};
    computeSus4(sus4 , dsus4, ddsus4, h0, sus40, hcrit);
    //computeSus4(sus4, dsus4, ddsus4, h_, sus40, hcrit);

    tensor<double,2> dvh_ = -deltat * ( outer(bf,Dh) + h * Dbf_g );


    tensor<double,1> dhh_ = -deltat * ( Dbf_g * v + bf * (divv + kd));
    tensor<double,3> dvdhh_ = -deltat * ( outer(bf,Dbf_g.T()) + outer(Dbf_g,bf) );
    tensor<double,3> dhdvh_ = dvdhh_.transpose({2,0,1});

    tensor<double,2> qbfvoigt = product(q,bf_voigt,{{0,2},{1,3}});
    //[5.1] RAYLEIGHIAN
    rayleighian += jac * h_ * (sus2*S2+sus4*S2*S2);
    // For plotting the global energies
    freeEnergyPotential += jac * h_ * (sus2*S2+sus4*S2*S2);
    //[5.2] GRADIENT
    Bk(all,range(1,2)) += 2.0 * (jac * h_ * (sus2 + 2.0 * sus4 * S2))  * qbfvoigt;
    Bk(all,range(3,4)) += (0.5 * jac * S2 * ( ( sus2 + sus4 * S2) + h_ * (dsus2 + dsus4 * S2))) * dvh_;


    //[5.3] HESSIAN
    Ak(all,range(1,2),all,range(1,2)) += 2.0 *  jac * h_ * ((sus2 + 2.0 * sus4 * S2) * product(bf_voigt,bf_voigt,{{2,2},{3,3}}) + 8.0 * sus4 * outer(qbfvoigt,qbfvoigt));
    Ak(all,range(1,2),all,range(3,4)) += 2.0 * jac * ((sus2 + 2.0 * sus4 * S2) + h_ * (dsus2 + 2.0 * dsus4 * S2)) * outer(qbfvoigt, dvh_);
    Ak(all,range(3,4),all,range(1,2)) += 2.0 * jac * ((sus2 + 2.0 * sus4 * S2) + h_ * (dsus2 + 2.0 * dsus4 * S2)) * outer(dvh_,qbfvoigt);
    Ak(all,range(3,4),all,range(3,4)) += (0.5 * jac * S2 * ( 2.0 * ( dsus2 + dsus4 * S2) + h_ * (ddsus2 + ddsus4 * S2))) * outer(dvh_,dvh_);
    Ak(all,range(1,2),all,0)          += 2.0 * jac * ((sus2 + 2.0 * sus4 * S2) + h_ * (dsus2 + 2.0 * dsus4 * S2)) * outer(qbfvoigt, dhh_);
    Ak(all,range(3,4),all,0)          += (0.5 * jac * S2 * ( 2.0 * ( dsus2 + dsus4 * S2) + h_ * (ddsus2 + ddsus4 * S2))) * outer(dvh_,dhh_) + (0.5 * jac * S2 * ( ( sus2 + sus4 * S2) + h_ * (dsus2 + dsus4 * S2))) *  dvdhh_;

    //-----------------------------------------------------------
    //[6] FREE ENERGY (FRANK)

    double DqDq = product(Dq,Dq,{{0,0},{1,1},{2,2}});
    tensor<double,2> DqDbfvoigt = product(Dq,Dbf_voigt,{{0,2},{1,3},{2,4}});
    //[6.1] RAYLEIGHIAN
    rayleighian += 0.5 * jac * h_* frank * DqDq;
    // For plotting the global energies

    freeEnergyPotential += 0.5 * jac * h_* frank * DqDq;
    //[6.2] GRADIENT
    Bk(all,range(1,2)) += (jac * h_ * frank) * DqDbfvoigt;
    Bk(all,range(3,4)) += (0.5 * jac * frank * DqDq) * dvh_;

    //[6.3] HESSIAN
    Ak(all,range(1,2),all,range(1,2)) += (jac * h_ * frank) * product(Dbf_voigt,Dbf_voigt,{{2,2},{3,3},{4,4}});
    Ak(all,range(1,2),all,range(3,4)) += (jac * frank) * outer(DqDbfvoigt,dvh_);
    Ak(all,range(3,4),all,range(1,2)) += (jac * frank) * outer(dvh_,DqDbfvoigt);

    Ak(all,range(1,2),all,0) += (jac * frank) * outer(DqDbfvoigt,dhh_);
    Ak(all,range(3,4),all,0) += (0.5 * jac * frank * DqDq) * dvdhh_;



    double rad = sqrt(x(0)*x(0) +  x(1)*x(1));
    //lambda_rot = lambda_rot*0.5*(1+tanh(10*(rad-1.1)));
    //lambda_aniso = lambda_aniso*0.5*(1+tanh(10*(rad-1.1)));
    //lambda_iso = lambda_iso*0.5*(1+tanh(10*(rad-1.1)));
    //-----------------------------------------------------------

    if (h0>0.5)
    {

        //        lambda_rot = 0.5*0.5*lambda_rot/(h0*h0);
        //        lambda_aniso = 0.5*lambda_aniso/h0;
    }

    // [7] POWER POTENTIAL[ISOTROPIC TERM]
    //[7.1]  RAYLEIGHIAN
    rayleighian += jac * h0 * lambda_iso* divv;
    // For plotting global power
    power +=  jac * h0 * lambda_iso* divv;
    double CF;

    if (1-sqrt(S2_0)<0)
        CF = 0.0;
    else
        CF = 1- sqrt(S2_0);

    //[7.2]  GRADIENT
    Bk(all,range(3,4)) += (jac * h0* lambda_iso*CF) * Dbf_g;

    //-----------------------------------------------------------

    // [8] POWER POTENTIAL[ANISOTROPIC TERM]
    tensor<double,2> S0nn0  = q0 + 0.5 * sqrt(S2_0)*id2d;
    tensor<double,2> S0nn0_ = -q0 + 0.5 * sqrt(S2_0)*id2d;
    //[8.1]  RAYLEIGHIAN
    rayleighian += (jac * h0 * lambda_aniso)* product(S0nn0,Dv,{{0,0},{1,1}});
    power +=  (jac * h0 * lambda_aniso)* product(S0nn0,Dv,{{0,0},{1,1}});
    //[8.2]  GRADIENT
    Bk(all,range(3,4)) += jac *( h0 * lambda_aniso) * (Dbf_g * S0nn0);
    Bk(all,range(3,4)) += (lambda_trans*h0)*jac * (Dbf_g*S0nn0_);


    //-----------------------------------------------------------
    // [9] POWER POTENTIAL [ROTATIONAL TERM]
    //[9.1]  RAYLEIGHIAN



    rayleighian += (jac * h0 * lambda_rot)* product(q0,Jq,{{0,0},{1,1}});
    // For plotting global power
    power +=  (jac * h0 * lambda_rot*h0)* product(q0,Jq,{{0,0},{1,1}});
    //[9.2]  GRADIENT
    Bk(all,range(1,2)) += (jac * h0 * lambda_rot*h0)* product(q0,dqJq_,{{0,2},{1,3}});
    Bk(all,range(3,4)) += (jac * h0 * lambda_rot*h0)* product(q0,dvJq_,{{0,2},{1,3}});

    Ak(all,range(1,2),all,range(3,4) )  += (jac * h0 * lambda_rot*h0)* product(q0,dvdqJq_,{{0,4},{1,5}});
    Ak(all,range(3,4),all,range(1,2) )  += (jac * h0 * lambda_rot*h0)* product(q0,dqdvJq_,{{0,4},{1,5}});


    //-----------------------------------------------------------
    // [10] SHEAR DISSIPATION
    rayleighian += (0.5 * jac * h0 * visc) * product(rodt,rodt,{{0,0},{1,1}});
    // For plotting global dissipations
    dissipation += (0.5 * jac * h0 * visc) * product(rodt,rodt,{{0,0},{1,1}});
    //[10.2] GRADIENT
    Bk(all,range(3,4)) += (jac * h0 * visc) * product(rodt,dvrodt,{{0,2},{1,3}});

    //[10.2] HESSIAN
    Ak(all,range(3,4),all,range(3,4)) +=  (jac * h0 * visc) * product(dvrodt,dvrodt,{{2,2},{3,3}});

    //-----------------------------------------------------------
    // [11] DILATATIONAL DISSIPATION (3D INCOMPRESSIBLE)
    rayleighian += (0.5 * jac * h0 * visc) * (divv * divv);
    // For plotting global dissipations
    dissipation +=  (0.5 * jac * h0 * visc) * (divv * divv);
    //[11.2] GRADIENT
    Bk(all,range(3,4)) += ((jac * h0 * visc) * divv) * Dbf_g;

    //[11.2] HESSIAN f
    Ak(all,range(3,4),all,range(3,4)) +=  (jac * h0 * visc) * outer(Dbf_g,Dbf_g);


    //rvisc  += 100*sqrt(S2_0)*rvisc; 
    //-----------------------------------------------------------
    //[12] ROTATIONAL DISSIPATION
    rayleighian += (0.5 * jac * h0 * rvisc) * product(Jq,Jq,{{0,0},{1,1}});
    //For plotting global dissipations
    dissipation += (0.5 * jac * h0 * rvisc) * product(Jq,Jq,{{0,0},{1,1}});
    //[12.2]  GRADIENT
    Bk(all,range(1,2)) += (jac * h0 * rvisc) * product(Jq_,dqJq,{{0,2},{1,3}});
    Bk(all,range(3,4)) += (jac * h0 * rvisc) * product(Jq_,dvJq_,{{0,2},{1,3}});

    //[12.3] HESSIAN
    Ak(all,range(1,2),all,range(1,2)) += (jac * h0 * rvisc) * product(dqJq,dqJq_,{{2,2},{3,3}});
    Ak(all,range(1,2),all,range(3,4)) += (jac * h0 * rvisc) * product(dqJq,dvJq_,{{2,2},{3,3}});


    Ak(all,range(3,4),all,range(1,2)) += (jac * h0 * rvisc) * product(dvJq_,dqJq_,{{2,2},{3,3}});
    Ak(all,range(3,4),all,range(1,2)) += (jac * h0 * rvisc) * product(Jq_,dqdvJq_,{{0,4},{1,5}}); 
    Ak(all,range(3,4),all,range(3,4)) += (jac * h0 * rvisc) * product(dvJq_,dvJq_,{{2,2},{3,3}});

    //-----------------------------------------------------------
    //[13] ROTATIONAL DISSIPATION
    rayleighian += (jac * h0 * cvisc) * product(rodt,Jq,{{0,0},{1,1}});
    //For plotting global dissipations
    dissipation += (jac * h0 * cvisc) * product(rodt,Jq,{{0,0},{1,1}});
    //[13.2]  GRADIENT
    Bk(all,range(1,2)) += (jac * h0 * cvisc) * product(rodt,dqJq,{{0,2},{1,3}});
    Bk(all,range(3,4)) += (jac * h0 * cvisc) * product(Jq,dvrodt,{{0,2},{1,3}});
    Bk(all,range(3,4)) += (jac * h0 * cvisc) * product(rodt,dvJq,{{0,2},{1,3}});

    //[13.3] HESSIAN
    Ak(all,range(1,2),all,range(3,4)) += (jac * h0 * cvisc) * product(dqJq,dvrodt,{{2,2},{3,3}});
    Ak(all,range(3,4),all,range(1,2)) += (jac * h0 * cvisc) * product(dvrodt,dqJq,{{2,2},{3,3}});
    Ak(all,range(3,4),all,range(3,4)) += (jac * h0 * cvisc) * product(dvrodt,dvJq,{{2,2},{3,3}});
    Ak(all,range(3,4),all,range(3,4)) += (jac * h0 * cvisc) * product(dvJq,dvrodt,{{2,2},{3,3}});




    //compLocalFric(fric, x(0),x(1));
    //-----------------------------------------------------------
    //[14] FRICTION
    rayleighian += (0.5 * jac * h0 * fric) * v * v;
    // For plotting global dissipations
    dissipation +=  (0.5 * jac * h0 * fric) * v * v;


    // [14.2] GRADIENT
    const double ramp = 3*std::max((phi - x_m), 0.0);
    const double coeff = jac * (h0 * fric + ramp);

    Bk(all, range(3,4)) += coeff * outer(bf, v);

    // [14.3] HESSIAN
    Ak(all, range(3,4), all, range(3,4)) +=coeff * outer(outer(bf, id2d), bf).transpose({0,1,3,2});
        

    //[14.2]  GRADIENT
    Bk(all,range(5,6)) += (jac * h0*0.1 ) * outer(bf,v_sub)/deltat;

    //[14.3] HESSIAN
    Ak(all,range(5,6),all,range(5,6)) += (jac * h0*0.1) * outer(outer(bf,id2d),bf).transpose({0,1,3,2})/(deltat*deltat);

    double fric_aniso{}; 
    //if (sqrt(S2_0)-0.6>0)
    {
        fric_aniso =  1E1; //*(sqrt(S2_0)-0.6);
    }
    {
        //[14] FRICTION
        rayleighian +=  (jac * fric_aniso)*(v-v_sub) * (v-v_sub);
        // For plotting global dissipations
        dissipation +=  (jac * fric_aniso)*(v-v_sub) * (v-v_sub);

        //[14.2]  GRADIENT
        //Bk(all,range(3,4)) += (jac * fric_aniso) * outer(bf,v-v_sub);
        Bk(all,range(5,6)) -= (jac * fric_aniso) * outer(bf,v-v_sub)/deltat;
        //[14.3] HESSIAN
        //Ak(all,range(3,4),all,range(3,4)) += (jac * fric_aniso) * outer(outer(bf,id2d),bf).transpose({0,1,3,2});
        //Ak(all,range(3,4),all,range(5,6)) -= (jac * fric_aniso) * outer(outer(bf,id2d),bf).transpose({0,1,3,2})/deltat;

        Ak(all,range(5,6),all,range(3,4)) -= (jac * fric_aniso) * outer(outer(bf,id2d),bf).transpose({0,1,3,2})/deltat;
        Ak(all,range(5,6),all,range(5,6)) += (jac * fric_aniso) * outer(outer(bf,id2d),bf).transpose({0,1,3,2})/(deltat*deltat);


    }

    //    if (sqrt(S2_0)>0.2)
    {

        Bk(all,range(3,4)) +=  (jac * h0 *1E3)*visc * product(S0nn0,dvrodt,{{0,2}, {1,3}} ) *product(S0nn0,rodt,{{0,0},{1,1}});
        Ak(all,range(3,4),all,range(3,4)) +=  (jac * h0 *1E3)*visc*outer(product(S0nn0,dvrodt,{{0,2}, {1,3}} ) ,product(S0nn0,dvrodt,{{0,2},{1,3}}));

    }

    //-----------------------------------------------------------
    //[15] CONSERVATION OF MASS
    double ledge = sqrt(jac);

    //    double modv;
    tensor<double,1> bf_supg = bf + stab * ledge* Dbf_g * v;
    //[15.2]  RHS
    Bk(all,0) += jac * (bf_supg * ( h - h_ ) + D*deltat*Dbf_g* Dh);

    //[15.3] GRADIENT (OF RHS)
    Ak(all,0,all,0)          += jac * (outer(bf_supg, ( bf - dhh_ )) + D*deltat*product(Dbf_g,Dbf_g,{{1,1}}));
    Ak(all,0,all,range(3,4)) += jac * outer(bf_supg, -dvh_ );
    Ak(all,0,all,range(3,4)) += (jac * (h - h_) * stab * ledge ) * outer(Dbf_g,bf).transpose({0,2,1});

    // Store global integrals
    //fillStr.addContribGlobInteg("dissipation", dissipation);


    tensor<double,2> T_upd = (nborCoords(all,range(0,1)) + nborDOFs0(all,range(5,6))).T() * Dbf_l;
    //Jacobian of the transformation (for integration)
    double jac_upd = T_upd.det();


    tensor<double,2> Du_sub = product(nbor_u_sub,Dbf_g,{{0,0}});
    tensor<double,4> dvDu_sub =(Dbf_g(K,n)*id2d(m,k))(K,k,m,n) ; 
    //Rate-of-deformation tensor
    tensor<double,2> strain_subs = 0.5*(Du_sub + Du_sub.transpose({1,0}));
    tensor<double,4> dvstrain_subs = 0.5 * ( outer(Dbf_g,id2d).transpose({0,3,1,2}) + outer(Dbf_g,id2d).transpose({0,3,2,1}) );



    Bk(all,range(5,6))(K,k) += (100*jac_upd) * (strain_subs(m,n)*dvstrain_subs(K,k,m,n)) ;
    Ak(all,range(5,6),all,range(5,6))(K,k,L,l) += (100*jac_upd) * dvstrain_subs(L,l,m,n)*dvstrain_subs(K,k,m,n) ; 
/*
    //Elastic energy per unit volume

    //Elastic energy per unit volume
    double rho = h;
    double rho0 = h0;
    double jac0 = jac;
    double lame_1 = 10/deltat; 
    double lame_2 = 50/deltat; 
    double eta = 1 + 1000*(1+tanh(100*(sqrt(S2_0) -0.7)));


    tensor<double,3> voigt_b={{{1,0},{0,0}},{{0,1},{1,0}},{{0,0},{0,1}}};
    tensor<double,3> voigt_Q = {{{1,0},{0,1}},
    {{0,1},{-1,0}}};

    //Basis functions times voigt tensor for interpolation of the finger tensor
    tensor<double,4> bfvoigt_b = bf(K)*voigt_b(l,m,n);
    tensor<double,4> bf_voigt_Q = outer(bf,voigt_Q.transpose({2,0,1}));
    //Gradient of the previous tensor (free index for derivation placed at the end)
    tensor<double,5> Dbf_g_voigt = outer(Dbf_g,voigt_b).transpose({0,2,3,4,1});
    tensor<double,5> Dbf_g_voigt_Q = outer(Dbf_g,voigt_Q.transpose({2,0,1})).transpose({0,2,3,4,1});

    tensor<double,2> d_divv = (dvDv(K,k,l,m)*id2d(l,m))(K,k);

    //previous time-step
    tensor<double,2> finger  =  (bfvoigt_b(K,l,m,n)*( nborDOFs(all,range(5,7))(K,l)  ))(m,n);
    tensor<double,2> finger0 =  (bfvoigt_b(K,l,m,n)*( nborDOFs0(all,range(5,7))(K,l) ))(m,n);
    tensor<double,2> ifinger0  = finger0.inv();
    tensor<double,2> ifinger   = finger.inv();


    double I = finger(k,l)*id2d(k,l);
    double J = sqrt(finger.det() * id2d.det());

    elasticmodel2D model;
    model.neohookean2(lame_1, lame_2, I, J);
    double elenergy = model.energy;
    double dI_elenergy = model.dI_energy;
    double dJ_elenergy = model.dJ_energy;
    double dII_elenergy = model.dII_energy;
    double dIJ_elenergy = model.dIJ_energy;
    double dJJ_elenergy = model.dJJ_energy;

    tensor<double,2> dfinger_I  = id2d;
    tensor<double,2> dfinger_J  = 0.5 * J * ifinger;     
    tensor<double,4> ddfinger_J = (0.25 * J * ifinger(k,l)*ifinger(m,n)-0.5 * J * ifinger(k,m)*ifinger(l,n))(k,l,m,n);

    tensor<double,2> dfinger_elenergy = dI_elenergy * dfinger_I(K,k) + dJ_elenergy * dfinger_J(K,k);

    tensor<double,4> ddfinger_elenergy = dII_elenergy * outer(dfinger_I,dfinger_I) + dIJ_elenergy * (outer(dfinger_I,dfinger_J)+outer(dfinger_J,dfinger_I))      + dJJ_elenergy * outer(dfinger_J,dfinger_J) + dJ_elenergy * ddfinger_J;

    tensor<double,2> df_elenergy =  dfinger_elenergy(K,k)*bfvoigt_b(L,l,K,k);  
    tensor<double,4> ddf_elenergy = product(bfvoigt_b,product(ddfinger_elenergy,bfvoigt_b,{{2,2},{3,3}}),{{2,0},{3,1}});


    tensor<double,1> Drho0_g  = nborDOFs0(all,0) * Dbf_g;
    tensor<double,1> Drho_g   = nborDOFs(all,0)  * Dbf_g;


    //[5.2] GRADIENT
    Bk(all,range(5,7))(K,k) =  (rho * jac) * df_elenergy(K,k);
    Bk(all,range(3,4))(K,k) +=   dvh_(K,k)*elenergy*jac;
    //[5.3] HESSIAN
    //finger - rho
    Ak(all,range(5,7),all,0)          +=  outer(jac * df_elenergy,bf);
    Ak(all,range(5,7),all,range(5,7)) +=  rho * jac * ddf_elenergy;


    Ak(all,range(3,4),all,0)(K,k,L)   +=  dvdhh_(K,k,L)*elenergy*jac;
    Ak(all,range(3,4),all,range(5,7))(K,k,L,l)   +=  dvh_(K,k)*df_elenergy(L,l)*jac;




    //[6] DISSIPATION

    //[6] DISSIPATION

    tensor<double,3> Dfinger0_g = product(nborDOFs0(all,range(5,7)),Dbf_g_voigt,{{0,0},{1,1}});
    tensor<double,2> lieFinger = (finger(k,l)-finger0(k,l))/deltat+ Dfinger0_g(k,l,m)*v(m)  - finger0(k,m)*Dv(l,m)  - Dv(k,m)*finger0(m,l) ;
    tensor<double,4> help1 =  (bf(K)*Dfinger0_g(k,l,m))(K,m,k,l);   
    tensor<double,4> help  =  (finger0(k,m)*dvDv(K,n,l,m))(K,n,k,l)  + (dvDv(K,n,k,m)*finger0(m,l))(K,n,k,l) ;
    //[6.1] RAYLEIGHIAN
    rayleighian += (eta/4.0 * jac0 * rho0) * product(lieFinger,lieFinger,{{0,0},{1,1}});
    //[6.2] GRADIENT
    Bk(all,range(5,7)) += (eta/2.0 * jac0 * rho0) * product(bfvoigt_b, lieFinger*ifinger0,{{2,0},{3,1}})/deltat;
    Bk(all,range(3,4)) += (eta/2.0 * jac0 * rho0) * product(help1,lieFinger*ifinger0,{{2,0},{3,1}});
    Bk(all,range(3,4)) -= (eta/2.0 * jac0 * rho0) * product(help, lieFinger*ifinger0,{{2,0},{3,1}});
    //[6.3] HESSIAN
    //b - all
    Ak(all,range(5,7),all,range(5,7)) += (eta/2.0 * jac0 * rho0) * product(bfvoigt_b,  bfvoigt_b*ifinger0,{{2,2},{3,3}})/(deltat*deltat);
    Ak(all,range(5,7),all,range(3,4)) += (eta/2.0 * jac0 * rho0) * product(bfvoigt_b,  help1*ifinger0,{{2,2},{3,3}})/deltat;
    Ak(all,range(5,7),all,range(3,4)) -= (eta/2.0 * jac0 * rho0) * product(bfvoigt_b,  help*ifinger0,{{2,2},{3,3}})/deltat;

    //[6] DISSIPATION      //v - all
    Ak(all,range(3,4),all,range(5,7)) += (eta/2.0 * jac0 * rho0) * product(help1,   bfvoigt_b*ifinger0,{{2,2},{3,3}})/deltat;
    Ak(all,range(3,4),all,range(3,4)) += (eta/2.0 * jac0 * rho0) * product(help1,   help1*ifinger0,{{2,2},{3,3}});
    Ak(all,range(3,4),all,range(3,4)) -= (eta/2.0 * jac0 * rho0) * product(help1,   help*ifinger0,{{2,2},{3,3}});

    Ak(all,range(3,4),all,range(5,7)) -= (eta/2.0 * jac0 * rho0) * product(help, bfvoigt_b*ifinger0,{{2,2},{3,3}})/deltat;
    Ak(all,range(3,4),all,range(3,4)) -= (eta/2.0 * jac0 * rho0) * product(help,   help1*ifinger0,{{2,2},{3,3}});
    Ak(all,range(3,4),all,range(3,4)) += (eta/2.0 * jac0 * rho0) * product(help,   help*ifinger0,{{2,2},{3,3}});

    */




}


void setSectorVelBC(SmartPtr<DOFsHandler> dofHand, SmartPtr<ParamStructure> userStr)
{
    using namespace hiperlife;

    for (int i = 0; i < dofHand->mesh->loc_nPts(); i++) 
    {

        double x = dofHand->mesh->nodeCoord(i, 0, IndexType::Local);
        double y = dofHand->mesh->nodeCoord(i, 1, IndexType::Local);
        int crease = dofHand->mesh->nodeCrease(i, IndexType::Local);
        double rad = sqrt(x * x + y * y);
        if (crease > 0)
        {
            if (rad<2)
            {
                  dofHand->nodeDOFs->setValue("vx", i, IndexType::Local, 0.0);
                  dofHand->setConstraint("vx", i, IndexType::Local, 0.0);
                  dofHand->nodeDOFs->setValue("vy", i, IndexType::Local, 0.0);
                  dofHand->setConstraint("vy", i, IndexType::Local, 0.0);
            }
        }
    }
}

std::vector<double> normalToEllipse(double x, double y, double a, double b) {

    double nx = 2 * x / (a * a);
    double ny = 2 * y / (b * b);

    // Calculate the magnitude of the gradient
    double magnitude = sqrt(nx * nx + ny * ny);

    // Normalize the components to get the unit normal vector
    std::vector<double> unitNormal = {nx / magnitude, ny / magnitude};
    
    return unitNormal;

}





void setSectorNemBC(SmartPtr<DOFsHandler> dofHand, SmartPtr<ParamStructure> userStr)
{
    using namespace hiperlife;
    using namespace std;
    for (int i = 0; i < dofHand->mesh->loc_nPts(); i++)
    {
        double x = dofHand->mesh->nodeCoord(i, 0, IndexType::Local);
        double y = dofHand->mesh->nodeCoord(i, 1, IndexType::Local);
        double rad = sqrt(x*x +y*y);
        double r_out = userStr->dparam[17] ;
        int crease = dofHand->mesh->nodeCrease(i, IndexType::Local);


        if (crease > 0)
        {
            double Q1{}, Q2{};
            double theeta =  atan(y / x);
            double angle =  std::atan(abs(y)/abs(x));
            if (angle < 0) 
            {
               angle += 2 * M_PI;  // Add 2π to make it positive
            }


   // Adjust the angle based on the quadrant
    if (x >= 0 && y >= 0) {
        // First quadrant, angle is correct
    } else if (x < 0 && y >= 0) {
        // Second quadrant
        angle = M_PI - angle;
    } else if (x < 0 && y < 0) {
        // Third quadrant
        angle = M_PI + angle;
    } else if (x >= 0 && y < 0) {
        // Fourth quadrant
        angle = 2 * M_PI - angle;
    }

            std::vector<double> normal = normalToEllipse(x, y, 8.3, 4.95);
            double n1 = normal[0]; 
            double n2 = normal[1];

            //double n1 = cos(theeta);
            //double n2 = sin(theeta);
            Q1 = n1*n1-0.5;
            Q2 = n1*n2;
            if (crease>0 and rad > 2)
            {
                double angle1 = 30.0; //false; //0.0 ; 
                double angle2 = 150.0;
                double angle3 = 210.0;  
                double angle4 = 330.0; //270.0; 
                double inc  = 2.0;    

                bool flag1 =  angle>(angle1-inc)*M_PI/180.0  and angle<(angle1+inc)*M_PI/180.0; 
                bool flag2 =  angle>(angle2-inc)*M_PI/180.0  and angle<(angle2+inc)*M_PI/180.0;
                bool flag3 =  angle>(angle3-inc)*M_PI/180.0  and angle<(angle3+inc)*M_PI/180.0;
                bool flag4 =  angle>(angle4-inc)*M_PI/180.0  and angle<(angle4+inc)*M_PI/180.0;
                if (false) //flag1 or flag2 or flag3 or flag4)
                {
                    dofHand->nodeDOFs->setValue(3, i, IndexType::Local, 0);
                    dofHand->nodeDOFs->setValue(4, i, IndexType::Local, 0);
                    dofHand->nodeDOFs->setValue(0, i, IndexType::Local, 0.5);
                    dofHand->nodeDOFs->setValue(1, i, IndexType::Local, Q1);
                    dofHand->nodeDOFs->setValue(2, i, IndexType::Local, Q2);

                    dofHand->setConstraint(3, i, IndexType::Local, 0.0);
                    dofHand->setConstraint(4, i, IndexType::Local, 0.0);
                    dofHand->setConstraint(0, i, IndexType::Local, 0.0); 
                    dofHand->setConstraint(1, i, IndexType::Local, 0.0);
                    dofHand->setConstraint(2, i, IndexType::Local, 0.0); 

                }
                if ( false) // theeta>-2.5*M_PI/180.0  and theeta<2.5*M_PI/180.0)
                {

                    dofHand->nodeDOFs->setValue(1, i, IndexType::Local, Q1);
                    dofHand->nodeDOFs->setValue(2, i, IndexType::Local, Q2);
                    dofHand->nodeDOFs->setValue(3, i, IndexType::Local, 0);
                    dofHand->nodeDOFs->setValue(4, i, IndexType::Local, 0);
                    dofHand->nodeDOFs->setValue(0, i, IndexType::Local, 0.5);

                    dofHand->setConstraint(1, i, IndexType::Local, 0.0);
                    dofHand->setConstraint(2, i, IndexType::Local, 0.0);
                    dofHand->setConstraint(3, i, IndexType::Local, 0.0);
                    dofHand->setConstraint(4, i, IndexType::Local, 0.0);
                    dofHand->setConstraint(0, i, IndexType::Local, 0.0);
                }
/*
                if (theeta>40*M_PI/180.0  and theeta<50*M_PI/180.0)
                {
                     dofHand->nodeDOFs->setValue(1, i, IndexType::Local, Q1);
                     dofHand->nodeDOFs->setValue(2, i, IndexType::Local, Q2);
                     dofHand->nodeDOFs->setValue(0, i, IndexType::Local, 0.5);
                     dofHand->setConstraint(1, i, IndexType::Local, 0.0);
                     dofHand->setConstraint(2, i, IndexType::Local, 0.0);
                     dofHand->setConstraint(0, i, IndexType::Local, 0.0);
                }
                if ( theeta<-40*M_PI/180.0  and theeta>-50*M_PI/180.0)
                {
                     dofHand->nodeDOFs->setValue(1, i, IndexType::Local, Q1);
                     dofHand->nodeDOFs->setValue(2, i, IndexType::Local, Q2);
                     dofHand->nodeDOFs->setValue(0, i, IndexType::Local, 0.5);
                     dofHand->setConstraint(1, i, IndexType::Local, 0.0);
                     dofHand->setConstraint(2, i, IndexType::Local, 0.0);
                     dofHand->setConstraint(0, i, IndexType::Local, 0.0);
                }
*/
            }
        }
        else
        {
            dofHand->nodeDOFs->setValue(1, i, IndexType::Local, 0.0);
            dofHand->nodeDOFs->setValue(2, i, IndexType::Local, 0.0);
        }
    }
}





void setSectorThiBC(SmartPtr<DOFsHandler> dofHand, SmartPtr<ParamStructure> userStr)
{
    using namespace hiperlife;
    double r_out = userStr->dparam[17] ;
    for (int i = 0; i < dofHand->mesh->loc_nPts(); i++)
    {
        double x = dofHand->mesh->nodeCoord(i, 0, IndexType::Local);
        double y = dofHand->mesh->nodeCoord(i, 1, IndexType::Local);
        double h_boundary = 0.2;
        dofHand->nodeDOFs->setValue(0, i, IndexType::Local, h_boundary);
        int crease = dofHand->mesh->nodeCrease(i, IndexType::Local);
        double rad = sqrt(x*x +y*y);

        if (crease > 0)
        {
            // In case of annulus making sure to apply BC only on the outer boundary for the outer boundary
            if (rad > 2)
            {
                dofHand->setConstraint(0, i, IndexType::Local, 0.0);
                //dofHand->setConstraint(5, i, IndexType::Local, 0.0);
                //dofHand->setConstraint(6, i, IndexType::Local, 0.0);
            }
            else
            {
                dofHand->setConstraint(5, i, IndexType::Local, 0.0);
                dofHand->setConstraint(6, i, IndexType::Local, 0.0);
            }

        }
    }

}



void setSectorConBC(SmartPtr<DOFsHandler> dofHand, SmartPtr<ParamStructure> userStr)
{
    using namespace hiperlife;
    double r_out = userStr->dparam[17] ;
    for (int i = 0; i < dofHand->mesh->loc_nPts(); i++)
    {
        double x = dofHand->mesh->nodeCoord(i, 0, IndexType::Local);
        double y = dofHand->mesh->nodeCoord(i, 1, IndexType::Local);
        double h_boundary = 0.2;
        dofHand->nodeDOFs->setValue(5, i, IndexType::Local, h_boundary);
        int crease = dofHand->mesh->nodeCrease(i, IndexType::Local);
        double rad = sqrt(x*x +y*y);

        if (crease > 0)
        {
            // In case of annulus making sure to apply BC only on the outer boundary for the outer boundary
            if (rad > r_out - 1e-4)
            {
                dofHand->setConstraint(5, i, IndexType::Local, 0.0);
            }
        }
    }

}


void setCircleVelBC(SmartPtr<DOFsHandler> dofHand, SmartPtr<ParamStructure> userStr)
{
    using namespace hiperlife;

    for (int i = 0; i < dofHand->mesh->loc_nPts(); i++)
    {
        double x = dofHand->mesh->nodeCoord(i, 0, IndexType::Local);
        double y = dofHand->mesh->nodeCoord(i, 1, IndexType::Local);
        // SETTING AUX-DOFs
        dofHand->nodeAuxF->setValue(0, i, IndexType::Local, x);
        dofHand->nodeAuxF->setValue(1, i, IndexType::Local, y);
        int crease = dofHand->mesh->nodeCrease(i, IndexType::Local);
        double r_in = userStr->dparam[16];
        //double r_out = userStr.dparam[17];
        double rad = sqrt(x * x + y * y);
        if (crease > 0 and rad<2.0)
        {
            double theeta = atan2(y, x);
            // theeta_ is the angle between 0 and 2pi
            double theeta_;
            //Since tan2 returns angles in interval [-pi,pi]
            if (theeta<0)
                theeta_ = theeta +2*M_PI;
            else
                theeta_ = theeta;
            double n1 = cos(theeta);
            double n2 = sin(theeta);
            double Vx, Vy, Vr;
            Vr = 5./60.0;
            //double slope = 5.0;
            //Components of velocity in cartesian coordiantes
            Vx = - Vr*n1;//*((0.5*(2-tanh(slope*((theeta_*180.0/M_PI)-85.0))+tanh(slope*((theeta_*180.0/M_PI)-95.0)))));
            Vy = - Vr*n2;//*((0.5*(2-tanh(slope*((theeta_*180.0/M_PI)-85.0))+tanh(slope*((theeta_*180.0/M_PI)-95.0)))));
                         //Radial velocity
                         //Prescribing the boundary conditions
            dofHand->nodeDOFs->setValue(3, i, IndexType::Local, 0.0);
            dofHand->nodeDOFs->setValue(4, i, IndexType::Local, 0.0);
            dofHand->setConstraint(3, i, IndexType::Local, 0.0);
            dofHand->setConstraint(4, i, IndexType::Local, 0.0);
        }
    }
}

void setSectorPhiBC(SmartPtr<DOFsHandler> dhand, SmartPtr<ParamStructure> userStr)
{
    using namespace hiperlife;
    for (int i = 0; i < dhand->mesh->loc_nPts(); i++)
    {
        double kon_phi  = userStr->dparam[26];
        double koff_phi = userStr->dparam[27];
        double x_m      = userStr->dparam[28];
        double mean  = kon_phi * x_m / (kon_phi + koff_phi); 
        double r2 = addGaussianNoise(mean, 0.01*mean);
        // Set the initial value with the sine wave perturbation
        dhand->nodeDOFs->setValue("phi", i, IndexType::Local,   r2);
        dhand->nodeDOFs0->setValue("phi", i, IndexType::Local,   r2);
    }
}


void LS_CortexNematic_Border(hiperlife::FillStructure& fillStr)
{
    using namespace hiperlife;
    using namespace hiperlife::Tensor;

    //-----------------------------------------------------------
    //[1] INPUT DATA
     auto& subFill = fillStr["cortexHand"];
    int nDim = subFill.nDim;
    int pDim = subFill.pDim;
    int eNN  = subFill.eNN;
    int numDOFs = subFill.numDOFs;
    int numAuxF = subFill.numAuxF;
    double uPoly = fillStr.paramStr->dparam[18];//5.0/60.0;
    double t_stall =  fillStr.paramStr->dparam[19]; //1.0
    double r_out = fillStr.paramStr->dparam[17] ;

    std::vector<double> tr = subFill.tangentsBoundaryRef();
    wrapper<double,1> tangentRef(tr.data(),2);   //2x1

    wrapper<double,2> nborCoords(subFill.nborCoords.data(),eNN,nDim);
    wrapper<double,2> nborDOFs0(subFill.nborDOFs0.data(),eNN,numDOFs);
    wrapper<double,2> nborDOFs(subFill.nborDOFs.data(),eNN,numDOFs);
    wrapper<double,2> nborAuxF(subFill.nborAuxF.data(),eNN,numAuxF);

    wrapper<double,1>  bf(subFill.getDer(0),eNN);
    wrapper<double,2> Dbf_l(subFill.getDer(1),eNN,pDim);
    wrapper<double,3> DDbf_l(subFill.getDer(2),eNN,pDim,pDim);
    //tensor<double,2> Dbf_g(eNN,pDim);
    tensor<double,3> DDbf_g(eNN,pDim,pDim);
    tensor<double,1> Lapbf_g(eNN);

    wrapper<double,2> Bk(fillStr.Bk(0).data(),eNN,numDOFs);
    wrapper<double,4> Ak(fillStr.Ak(0,0).data(),eNN,numDOFs,eNN,numDOFs);

    //[2] GEOMETRY
    tensor<double,1> x = bf * nborCoords;
    //Transformation matrix (from reference to spatial)
    tensor<double,2> T = nborCoords(all,range(0,1)).T() * Dbf_l; // x_{aI} DN_{I1}
    tensor<double,1> tangent = T * tangentRef; // a*
    double normt = sqrt(tangent(0)*tangent(0)+tangent(1)*tangent(1));
    tensor<double,1> bnormal = {tangent(1),-tangent(0)};
    bnormal /= normt;
    //Global derivatives (wrt x and y ) of the basis functions
    double jac;
    tensor<double,2>  Dbf_g = Dbf_l*T.inv();

    //GlobalBasisFunctions::hessians(Dbf_g, jac, DDbf_g, Lapbf_g, eNN, pDim, nDim, nborCoords, Dbf_l, DDbf_l);
    //[3] VARIABLES

    //[3.1] AUXILIARY
    tensor<double,1> nbor_h  = nborDOFs(all,0);
    tensor<double,1> nbor_h0 = nborDOFs0(all,0);
    tensor<double,2> nbor_q  = nborDOFs(all,range(1,2));
    tensor<double,2> nbor_q0 = nborDOFs0(all,range(1,2));
    tensor<double,2> nbor_v = nborDOFs(all,range(3,4));

    tensor<double,2> id2d = {{1.0,0.0},{0.0,1.0}};
    tensor<double,3> voigt = {{{1,0},{0,1}},
                              {{0,1},{-1,0}}};
    tensor<double,4> bf_voigt = outer(bf,voigt.transpose({2,0,1}));
    tensor<double,5> Dbf_voigt = outer(Dbf_g,voigt.transpose({2,0,1})).transpose({0,2,3,4,1});

    //[3.2] STATE/ PROCESS VARIABLES
    double h0 = bf * nbor_h0;
    tensor<double,1> v  = bf * nbor_v;
    double vn = v*bnormal;
    double vp  = uPoly;
    double rad=sqrt(x(0)*x(0) + x(1)*x(1));
    if (rad>r_out-0.1)
    {

        double rayleighian{};
        //-----------------------------------------------------------
        //[1.1]  Rayleighian;
        rayleighian += normt *h0* (0.5 * t_stall / uPoly) * (vn - vp) * (vn - vp);
        //[1.2] GRADIENT
        //Bk(all, range(3, 4)) += normt *h0* (t_stall / uPoly) * (vn - vp) * outer(bf, bnormal);
        //[1.3] Hessian
        //Ak(all, range(3, 4), all, range(3, 4)) += normt *h0* (t_stall / uPoly) * outer(outer(bf, bnormal), outer(bf, bnormal));

    }

}


void setLinearConstraints(std::vector<double>& dparam, Teuchos::RCP<hiperlife::HiPerProblem> hiperProbl)
{
    using namespace hiperlife;
    for (int i = 0; i < hiperProbl->_dhands[0]->mesh->_linearConstraints.size(); i++)
    {
        int master = hiperProbl->_dhands[0]->mesh->_linearConstraints[i].masters[0];
        int slave = hiperProbl->_dhands[0]->mesh->_linearConstraints[i].slave;
        double theeta= dparam[15];
        // Transformation for h is already done inside hiperlife
        // Transformation for Qs
        double wgtq_1 = cos(2*theeta);
        double wgtq_2 = sin(2*theeta);
        ////For Q1
        hiperProbl->setLinearConstraint({0, 1, slave, IndexType::Global}, {0, 1, master, IndexType::Global},wgtq_1-1);
        hiperProbl->setLinearConstraint({0, 1, slave, IndexType::Global}, {0, 2, master, IndexType::Global},-wgtq_2);
        ////For Q2
        hiperProbl->setLinearConstraint({0, 2, slave, IndexType::Global}, {0, 2, master, IndexType::Global},wgtq_1-1);
        hiperProbl->setLinearConstraint({0, 2, slave, IndexType::Global}, {0, 1, master, IndexType::Global},wgtq_2);

        // Transformation for velocities
        double wgtV_1 = cos(theeta);
        double wgtV_2 = sin(theeta);
        //// Transformation of Vx
        //// For vx
        hiperProbl->setLinearConstraint({0, 3, slave, IndexType::Global}, {0, 3, master, IndexType::Global},wgtV_1-1);
        hiperProbl->setLinearConstraint({0, 3, slave, IndexType::Global}, {0, 4, master, IndexType::Global},-wgtV_2);
        //// For vy
        hiperProbl->setLinearConstraint({0, 4, slave, IndexType::Global}, {0, 4, master, IndexType::Global},wgtV_1-1);
        hiperProbl->setLinearConstraint({0, 4, slave, IndexType::Global}, {0, 3, master, IndexType::Global},wgtV_2);
    }

}




