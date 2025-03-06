/// BREMCANNONSimVisManager.hh
//// Auteur: Arnaud HUBER for ENL group <huber@cenbg.in2p3.fr>
//// Copyright: 2022 (C) Projet RATP - ENL [LP2IB] - CELIA

#ifndef BREMCANNONSimVisManager_h
#define BREMCANNONSimVisManager_h 1

#include "G4VisManager.hh"


class BREMCANNONSimVisManager: public G4VisManager {

public:

  BREMCANNONSimVisManager ();

private:

  void RegisterGraphicsSystems ();

};

#endif
