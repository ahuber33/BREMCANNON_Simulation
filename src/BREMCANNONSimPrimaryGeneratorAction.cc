/// BREMCANNONSimPrimaryGeneratorAction.cc
//// Auteur: Arnaud HUBER for ENL group <huber@cenbg.in2p3.fr>
//// Copyright: 2022 (C) Projet RATP - ENL [LP2IB] - CELIA

#include "BREMCANNONSimPrimaryGeneratorAction.hh"


G4UImanager* UI = G4UImanager::GetUIpointer();
BREMCANNONSimPrimaryGeneratorAction::BREMCANNONSimPrimaryGeneratorAction(){
  //G4int n_particle = 1;

  //particleGun = new G4ParticleGun(n_particle);
  //UI->ApplyCommand("/control/execute setgun.mac");
  //UI->ApplyCommand("/control/execute setgun_script_use.mac");

  particleSource = new G4GeneralParticleSource();
  //UI->ApplyCommand("/control/execute setsource.mac");
}

BREMCANNONSimPrimaryGeneratorAction::~BREMCANNONSimPrimaryGeneratorAction(){
  //delete particleGun;
  delete particleSource;
}

void BREMCANNONSimPrimaryGeneratorAction::GeneratePrimaries(G4Event* anEvent){

  //particleGun->GeneratePrimaryVertex(anEvent);
  particleSource->GeneratePrimaryVertex(anEvent);

  //  always require these two lines
  G4EventManager *evtman = G4EventManager::GetEventManager();
  BREMCANNONSimEventAction *evtac = (BREMCANNONSimEventAction*)evtman->GetUserEventAction();

  //evtac->SetIncidentE(particleGun->GetParticleEnergy());
  //evtac->SetIncidentE(particleSource->GetParticleEnergy());
}
