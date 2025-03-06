/// BREMCANNONSimPhysics.hh
//// Auteur: Arnaud HUBER for ENL group <huber@cenbg.in2p3.fr>
//// Copyright: 2022 (C) Projet RATP - ENL [LP2IB] - CELIA

#ifndef  BREMCANNONSimPhysics_h
#define   BREMCANNONSimPhysics_h  1

//#include "G4VUserPhysicsList.hh"
#include "G4VModularPhysicsList.hh"
#include "CLHEP/Units/SystemOfUnits.h"
#include "G4DecayPhysics.hh"
#include "G4RadioactiveDecayPhysics.hh"
#include "G4EmStandardPhysics_option3.hh"

class  BREMCANNONSimPhysics:  public G4VModularPhysicsList
{      public:
  BREMCANNONSimPhysics();
  virtual ~BREMCANNONSimPhysics();

protected:
  virtual void ConstructParticle();
  virtual void ConstructProcess();
  virtual void SetCuts();

private:
  G4VPhysicsConstructor*  emPhysicsList;
  G4VPhysicsConstructor*  particleList;
  G4VPhysicsConstructor*  raddecayList;
};
#endif
