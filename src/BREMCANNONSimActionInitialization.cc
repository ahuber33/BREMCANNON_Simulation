/// BREMCANNONSimRunAction.cc
//// Auteur: Arnaud HUBER for ENL group <huber@lp2ib.in2p3.fr>
//// Copyright: 2024 (C) Projet PALLAS

#include "BREMCANNONSimActionInitialization.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

BREMCANNONSimActionInitialization::BREMCANNONSimActionInitialization(BREMCANNONSimGeometry* Geom, const char *suff, G4bool pMT)
    : G4VUserActionInitialization(), 
      detector(Geom),
      suffixe(suff),
      flag_MT(pMT)
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

BREMCANNONSimActionInitialization::~BREMCANNONSimActionInitialization()
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void BREMCANNONSimActionInitialization::BuildForMaster() const
{
    // Action pour le processus maître (uniquement utile en mode multithreading)
    SetUserAction(new BREMCANNONSimRunAction(detector, suffixe, flag_MT));
    G4cout << "Build Master" << G4endl;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void BREMCANNONSimActionInitialization::Build() const
{
    // Création et affectation des actions pour la simulation
    auto *runAction = new BREMCANNONSimRunAction(detector, suffixe, flag_MT);
    auto *eventAction = new BREMCANNONSimEventAction(suffixe);
    
    // Assigner les actions utilisateur
    SetUserAction(new BREMCANNONSimPrimaryGeneratorAction());
    SetUserAction(runAction);
    SetUserAction(eventAction);
    SetUserAction(new BREMCANNONSimSteppingAction(detector, runAction, eventAction));
    //SetUserAction(new BREMCANNONSimTrackingAction());
}