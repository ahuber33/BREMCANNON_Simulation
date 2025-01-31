/// BREMCANNONSimPrimaryGeneratorAction.hh
//// Auteur: Arnaud HUBER for ENL group <huber@cenbg.in2p3.fr>
//// Copyright: 2022 (C) Projet RATP - ENL [LP2IB] - CELIA

#ifndef BREMCANNONSimPrimaryGeneratorAction_h
#define BREMCANNONSimPrimaryGeneratorAction_h 1

#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4ParticleGun.hh"
#include "G4GeneralParticleSource.hh"
#include "G4UImanager.hh"
#include "G4EventManager.hh"
#include "BREMCANNONSimEventAction.hh"

class G4Event;

class BREMCANNONSimPrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
public:
  BREMCANNONSimPrimaryGeneratorAction();
  ~BREMCANNONSimPrimaryGeneratorAction();

public:
  void GeneratePrimaries(G4Event* anEvent);
  void SetEnergy(G4double en){particleGun->SetParticleEnergy(en);};

  G4double GetEnergy(){return particleGun->GetParticleEnergy();};

private:
  G4ParticleGun* particleGun;
  G4GeneralParticleSource *particleSource;
  G4double IncidentE;
};

#endif
