#ifndef AuxCortexNematic2D
#define AuxCortexNematic2D

#include <Teuchos_RCP.hpp>
#include <Teuchos_CommandLineProcessor.hpp>

#include "hl_FillStructure.h"
#include "hl_HiPerProblem.h"
#include "hl_Tensor.h"

#include "hl_DOFsHandler.h"

using namespace hiperlife;
using namespace hiperlife::Tensor;



void LS_CortexNematic_2D(hiperlife::FillStructure& fillStr);
void LS_phi(hiperlife::FillStructure& fillStr);
void LS_border_phi(hiperlife::FillStructure& fillStr);

int CortexMechanicsSolver (std::vector<double>& , std::vector<int>& , Teuchos::RCP<hiperlife::HiPerProblem> );
void setSectorVelBC(SmartPtr<DOFsHandler> dofHand, SmartPtr<ParamStructure> userStr);
void setSectorNemBC(SmartPtr<DOFsHandler> dofHand, SmartPtr<ParamStructure> userStr);
void setSectorThiBC(SmartPtr<DOFsHandler> dofHand, SmartPtr<ParamStructure> userStr);
void setCircleVelBC(SmartPtr<DOFsHandler> dofHand, SmartPtr<ParamStructure> userStr);
void setSectorConBC(SmartPtr<DOFsHandler> dofHand, SmartPtr<ParamStructure> userStr);
void setSectorPhiBC(SmartPtr<DOFsHandler> dhand, SmartPtr<ParamStructure> userStr);
void setLinearConstraints(std::vector<double>& dparam, Teuchos::RCP<hiperlife::HiPerProblem> hiperProbl);
void LS_CortexNematic_Border(hiperlife::FillStructure& fillStr);
inline void computeSus2(double &sus2, double &dsus2, double &ddsus2, double h, double sus20, double hcrit)
{
    sus2  = sus20* hcrit;// * (hcrit - h);
    dsus2 = 0.0;//;
    ddsus2 = 0.0;
}

inline void computeSus4(double &sus4, double &dsus4, double &ddsus4, double h, double sus40, double hcrit)
{

    sus4  = sus40 * h;
    dsus4 = 0.0;
    ddsus4 = 0.0;
}




inline double  computekd(double kd0, double S0 )
{
    return kd0;
}

// Defining a spatial field for the polymerization and de-polymerization rate


inline void compTurnOverSpaFld_mec1(double &kp, double &kd, tensor<double,1> &dkp, tensor<double,1> &dkd,double heqb, double kd0,tensor<double,1> x)
{


    // Radius of the region with high turn-over

    double radius    = 2.0;
    double slope     = 100.0;
    // Values of the over coefficients in the high polymerization region
    double scale_up_kd  = 20.;
    double scale_down   = kd0;
    double thick_equip  = heqb;
    double r_ref         = sqrt( pow(x(0),2) + pow(x(1),2));
    tensor<double,1> dr_ref = x/r_ref;

    // TURN-OVER COEFFICIENTS
    kp=thick_equip*(scale_up_kd*0.5*(1-tanh(slope*(r_ref-radius)))+scale_down);
    kd=scale_up_kd*0.5*(1-tanh(slope*(r_ref-radius)))+scale_down;

    //DERIVATIVES OF TURN-OVER COEFFICIENTS
    dkp =  -thick_equip*(scale_up_kd*0.5*(1-tanh(slope*(r_ref-radius))*tanh(slope*(r_ref-radius)))*slope*dr_ref); //FIXME:: Maybe I dont have to use it at all
    dkd =  -scale_up_kd*0.5*(1-tanh(slope*(r_ref-radius))*tanh(slope*(r_ref-radius)))*slope*dr_ref;


}



inline void compTurnOverSpaFld_mec2(double &kp, double &kd, tensor<double,1> &dkp, tensor<double,1> &dkd,double heqb, double kd0,tensor<double,1> x)
{


    // Radius of the region with high turn-over
    double radius    = 1.25;
    double slope     = 2.0;
    // Values of the over coefficients in the high polymerization region
    double scale_up_kd  = 2000.0;
    double scale_down  = kd0;
    double thick_equip = heqb;
    double r_ref = sqrt( pow(x(0),2) + pow(x(1),2));
    tensor<double,1> dr_ref = x/r_ref;

    // TURN-OVER COEFFICIENTS
    kp=thick_equip*(scale_up_kd*0.5*(1-tanh(slope*(r_ref-radius))))+ kp;
    kd=scale_up_kd*0.5*(1-tanh(slope*(r_ref-radius))) + kd;
    //DERIVATIVES OF TURN-OVER COEFFICIENTS
    dkp =  0.0;//-thick_equip*(scale_up_kd*0.5*(1-tanh(slope*(r_ref-radius))*tanh(slope*(r_ref-radius)))*slope*dr_ref); //FIXME:: Maybe I dont have to use it at all
    dkd =  0.0;//-scale_up_kd*0.5*(1-tanh(slope*(r_ref-radius))*tanh(slope*(r_ref-radius)))*slope*dr_ref;

}


// Defining a spatial field for friction
inline void compTurnOverFricFld(double &Friction, double xcoord, double ycoord)
{
    double xcoordRef = 4.7;
    double ycoordRef = 0.0;
    //radius of the stress fibers
    double radius    = 0.2;
    Friction=(Friction-20)*0.5*(1-tanh(5*(sqrt(pow((xcoordRef - xcoord),2) + pow((ycoordRef - ycoord),2))-radius)))+20;
    return;
}


inline void compLocalFric(double &Friction, double xCoord, double yCoord)
{

   double rad = sqrt(xCoord*xCoord + yCoord*yCoord);
   double slope = 100;
   Friction = 100*Friction*0.5*(tanh(slope*(rad-2.75)) - tanh(slope*(rad-3.0)));
}




#endif //AuxCortexNematic2D




