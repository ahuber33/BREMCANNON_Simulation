/// BREMCANNONSimSteppingAction.cc
//// Auteur: Arnaud HUBER for ENL group <huber@cenbg.in2p3.fr>
//// Copyright: 2022 (C) Projet RATP - ENL [LP2IB] - CELIA

#ifndef BREMCANNONSimSteppingAction_h
#define BREMCANNONSimSteppingAction_h

#include "G4UserSteppingAction.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4ProcessManager.hh"
#include "G4Track.hh"
#include "BREMCANNONSimRunAction.hh"


class G4Step;
class G4SteppingManager;
class BREMCANNONSimGeometry;
class BREMCANNONSimRunAction;
class BREMCANNONSimEventAction;

class BREMCANNONSimSteppingAction : public G4UserSteppingAction
{
public:
  BREMCANNONSimSteppingAction(BREMCANNONSimGeometry*,BREMCANNONSimRunAction*,BREMCANNONSimEventAction*);
  ~BREMCANNONSimSteppingAction();
public:
  void UserSteppingAction(const G4Step*);

private:
  BREMCANNONSimGeometry * detector;
  BREMCANNONSimRunAction* runaction;
  BREMCANNONSimEventAction* eventaction;
  static const G4String path;
  G4double PosZ_Sensitive[20];
  G4int Parent_ID=0;
  G4int StepNo=0;
  G4Track* track=nullptr;
  G4StepPoint* thePrePoint=nullptr;
  G4StepPoint* thePostPoint=nullptr;
  G4String ProcessName="";
  G4String VolumName="";
  G4String PreVolumName="";
  G4String PartName="";
  G4int ID=0;
  G4ThreeVector prePoint;
  G4ThreeVector postPoint;
  G4ThreeVector point;
  G4double x=0.0;
  G4double y=0.0;
  G4double z=0.0;
  G4double Edep=0.0;
  G4double EdepEff=0.0;
  G4double Energy=0.0;
  G4double Labs=0.0;
  G4int Number_IP=0;

};
#endif
