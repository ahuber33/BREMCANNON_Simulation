/// BREMCANNONSimGeometry.hh
//// Auteur: Arnaud HUBER for ENL group <huber@cenbg.in2p3.fr>
//// Copyright: 2022 (C) Projet RATP - ENL [LP2IB] - CELIA

#ifndef BREMCANNONSimGeometry_h
#define BREMCANNONSimGeometry_h 1

#include "G4VUserDetectorConstruction.hh"
#include "BREMCANNONSimMaterials.hh"
#include "Geometry.hh"
#include "G4VisAttributes.hh"
#include "G4Colour.hh"

class Geometry;
class BREMCANNONSimMaterials;
class G4VPhysicalVolume;

class BREMCANNONSimGeometry : public G4VUserDetectorConstruction
{
public:
  BREMCANNONSimGeometry();
  ~BREMCANNONSimGeometry();

public:
  G4VPhysicalVolume *Construct();
  void SetLogicalVolumeColor(G4LogicalVolume*, G4String);
  void SetLogicalVolumeColorFromMaterial(G4LogicalVolume*, G4Material*);
  void ConstructWorld();
  void ConstructFiltersAndIPs();
  void ConstructCylinderPart(G4int);
  void GetVariables();
  void ConstructMaterialsList();
  G4double GetRadiusCylinderInternal() { return Radius_cylinder_internal; }
  G4double GetRadiusCylinderExternal() { return Radius_cylinder_external; }
  G4double GetStackIPPosIni() { return Stack_IP_pos_ini; }
  G4double GetDiskThickness() { return Disk_thickness; }
  G4double GetMylarThickness() { return Mylar_thickness; }
  G4double GetFlagMylar() { return flag_Mylar; }
  G4double GetCylinderTantaleLength() { return Cylinder_tantale_length; }
  G4double GetNumberIP() { return Number_IP; }
  G4double GetIPa1z() { return IPa1_z; }
  G4double GetIPa2z() { return IPa2_z; }
  G4double GetIPa3z() { return IPa3_z; }
  G4double GetIPa4z() { return IPa4_z; }
  G4double GetThicknessFilter(int n) { return Thickness_Filter[n]; }
  G4double GetPosZSensitive(int n) { return PosZSensitive[n]; }
  G4double GetLabs() { return Labs; }
  G4double GetCoefPSL() { return Coef_PSL; }

private:
  static const G4String path_bin;

  Geometry *Geom;

    // Colors for visualizations
  G4VisAttributes *invis;
  G4VisAttributes *white;
  G4VisAttributes *gray;
  G4VisAttributes *black;
  G4VisAttributes *red;
  G4VisAttributes *orange;
  G4VisAttributes *yellow;
  G4VisAttributes *green;
  G4VisAttributes *cyan;
  G4VisAttributes *blue;
  G4VisAttributes *magenta;

  // Logical Volumes
  G4LogicalVolume *LogicalWorld=nullptr;
  G4LogicalVolume *LogicalFilter[20];

  // Physical volumes
  G4VPhysicalVolume *PhysicalWorld=nullptr;
  G4VPhysicalVolume *physiStackIP1[20];
  G4VPhysicalVolume *physiStackIP2[20];
  G4VPhysicalVolume *physiStackIP3[20];
  G4VPhysicalVolume *physiStackIP4[20];
  G4VPhysicalVolume *physiDisk[20];

  G4Material* Alu;
  G4Material* Cuivre;
  G4Material* Etain;
  G4Material* Plomb;
  G4Material* Titane;
  G4Material* Fer;
  G4Material* Tantale;
  
  G4double Radius_cylinder_internal=0.0;
  G4double Radius_cylinder_external=0.0;
  G4double Radius_IP=0.0;
  G4double Stack_IP_pos_ini=0.0;
  G4double Disk_thickness=0.0;
  G4double Mylar_thickness=0.0;
  G4int flag_Mylar=1;
  G4double Cylinder_tantale_length=0.0;
  G4double Zposition=0.0;
  G4int Number_IP=0;
  G4double IPa1_z=0.0;
  G4double IPa2_z=0.0;
  G4double IPa3_z=0.0;
  G4double IPa4_z=0.0;
  G4double Thickness_Filter[20];
  G4double PosZSensitive[20];
  G4String Name_Material_Filter[20];
  G4Material* Material_Filter[20];
  G4double Labs=0.0;
  G4double Coef_PSL=0.0;
};
#endif
