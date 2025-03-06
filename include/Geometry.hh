/// Geometry.hh
//// Auteur: Arnaud HUBER for ENL group <huber@cenbg.in2p3.fr>
//// Copyright: 2022 (C) Projet RATP - ENL [LP2IB] - CELIA

#ifndef Geometry_h
#define Geometry_h

#include "G4LogicalVolume.hh"
#include "CLHEP/Units/SystemOfUnits.h"
#include "G4Transform3D.hh"
#include "G4UnionSolid.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4Cons.hh"
#include "G4Polyhedra.hh"
#include <G4Polycone.hh>
#include "G4Sphere.hh"
#include "G4Trap.hh"
#include "G4Trd.hh"
#include "G4SubtractionSolid.hh"
#include "G4PVPlacement.hh"
#include "G4UnitsTable.hh"
#include <math.h>

// #ifndef disable_gdml
#include "G4GDMLParser.hh"
// #endif

class Geometry
{
public:

  //constructor, builds from keys specified in buildfile
  Geometry(G4String buildfile);
  ~Geometry();
  //  void Construct();

public:
  G4LogicalVolume *GetFilter(G4String, G4double, G4double, G4Material*);
  G4LogicalVolume *GetStackIP1(G4String, G4Material*);
  G4LogicalVolume *GetStackIP2(G4String, G4Material*);
  G4LogicalVolume *GetStackIP3(G4String, G4Material*);
  G4LogicalVolume *GetStackIP4(G4String, G4Material*);
  G4LogicalVolume *GetCorpus(G4String, G4Material*);
  G4LogicalVolume *GetVolumeFiller(G4String, G4Material*, G4double);
  G4LogicalVolume *GetFullDisk(G4String, G4Material*);
  G4LogicalVolume *GetOpenDisk(G4String, G4Material*);
  G4LogicalVolume *GetEntryProtection(G4String, G4Material*);
  G4LogicalVolume *GetPlaqueTantale(G4String, G4Material*);

  G4double GetRadiusCylinderInternal(){return Radius_cylinder_internal;}
  G4double GetRadiusCylinderExternal(){return Radius_cylinder_external;}
  G4double GetStackIPPosIni(){return Stack_IP_pos_ini;}
  G4double GetDiskThickness(){return Disk_thickness;}
  G4double GetMylarThickness(){return Mylar_thickness;}
  G4int GetFlagMylar(){return flag_Mylar;}
  G4double GetCylinderTantaleLength(){return Cylinder_tantale_length;}
  G4double GetNumberIP(){return Number_IP;}
  G4double GetRadiusIP(){return Radius_IP;}
  G4double GetIPa1z(){return IPa1_z;}
  G4double GetIPa2z(){return IPa2_z;}
  G4double GetIPa3z(){return IPa3_z;}
  G4double GetIPa4z(){return IPa4_z;}
  G4double GetThicknessFilter(int n){return Thickness_Filter[n];}
  G4String GetMaterialFilter(int n){return Material_Filter[n];}
  G4double GetLabs(){return Labs;}
  G4double GetCoefPSL(){return Coef_PSL;}


private:
  Geometry *Geom;

  static const G4String path_bin;

  // Logical Volumes
  G4LogicalVolume *LogicalVolume;

  G4double Radius_cylinder_internal=0.0;
  G4double Radius_cylinder_external=0.0;
  G4double Radius_IP=0.0;
  G4double Stack_IP_pos_ini=0.0;
  G4double Disk_thickness=0.0;
  G4double Mylar_thickness=0.0;
  G4int flag_Mylar=0;
  G4double Cylinder_tantale_length=0.0;
  G4int Number_IP=0;
  G4double IPa1_z=0.0;
  G4double IPa2_z=0.0;
  G4double IPa3_z=0.0;
  G4double IPa4_z=0.0;
  G4double Thickness_Filter[20];
  G4String Material_Filter[20];
  G4double Labs=0.0;
  G4double Coef_PSL=0.0;

  // Other
  G4VisAttributes *clear;

};
#endif
