// BREMCANNONSimGeometry_test.cc
//// Auteur: Arnaud HUBER for ENL group <huber@cenbg.in2p3.fr>
/// Copyright: 2017 (C) Projet BADGE - CARMELEC -P2R

#include "BREMCANNONSimGeometry.hh"

using namespace CLHEP;

const G4String BREMCANNONSimGeometry::path_bin = "../bin/";

// Constructor
BREMCANNONSimGeometry::BREMCANNONSimGeometry()
{}

// Destructor
BREMCANNONSimGeometry::~BREMCANNONSimGeometry()
{}

void BREMCANNONSimGeometry::SetLogicalVolumeColor(G4LogicalVolume *LogicalVolume, G4String Color)
{
  // ***********************
  // Visualization Colors
  // ***********************
  // Create some colors for visualizations
  invis = new G4VisAttributes(G4Colour(255 / 255., 255 / 255., 255 / 255.));
  invis->SetVisibility(false);

  white = new G4VisAttributes(G4Colour(1, 1, 1, 0.1)); // Sets the color (can be looked up online)
  // white->SetForceWireframe(true); // Sets to wire frame mode for coloring the volume
  white->SetForceSolid(true); // Sets to solid mode for coloring the volume
  white->SetVisibility(true); // Makes color visible in visualization

  gray = new G4VisAttributes(G4Colour(0.5, 0.5, 0.5, 1));
  //  gray->SetForceWireframe(true);
  gray->SetForceSolid(true);
  gray->SetVisibility(true);

  black = new G4VisAttributes(G4Colour(0, 0, 0, 1));
  //  black->SetForceWireframe(true);
  black->SetForceSolid(true);
  black->SetVisibility(true);

  red = new G4VisAttributes(G4Colour(1, 0, 0, 1));
  //  red->SetForceWireframe(true);
  red->SetForceSolid(true);
  red->SetVisibility(true);

  orange = new G4VisAttributes(G4Colour(1, 0.5, 0, 1));
  //  orange->SetForceWireframe(true);
  orange->SetForceSolid(true);
  orange->SetVisibility(true);

  yellow = new G4VisAttributes(G4Colour(1, 1, 0, 0.29));
  //  yellow->SetForceWireframe(true);
  yellow->SetForceSolid(true);
  yellow->SetVisibility(true);

  green = new G4VisAttributes(G4Colour(0, 1, 0, 0.35));
  //  green->SetForceWireframe(true);
  green->SetForceSolid(true);
  green->SetVisibility(true);

  cyan = new G4VisAttributes(G4Colour(0, 1, 1, 0.55));
  //  cyan->SetForceWireframe(true);
  cyan->SetForceSolid(true);
  cyan->SetVisibility(true);

  blue = new G4VisAttributes(G4Colour(0, 0, 1, 1));
  //  blue->SetForceWireframe(true);
  blue->SetForceSolid(true);
  blue->SetVisibility(true);

  magenta = new G4VisAttributes(G4Colour(1, 0, 1, 0.85));
  //  magenta->SetForceWireframe(true);
  // magenta->SetForceSolid(true);
  magenta->SetVisibility(true);

  if (Color == "invis")
  {
    LogicalVolume->SetVisAttributes(invis);
  }
  else if (Color == "black")
  {
    LogicalVolume->SetVisAttributes(black);
  }
  else if (Color == "white")
  {
    LogicalVolume->SetVisAttributes(white);
  }
  else if (Color == "gray")
  {
    LogicalVolume->SetVisAttributes(gray);
  }
  else if (Color == "red")
  {
    LogicalVolume->SetVisAttributes(red);
  }
  else if (Color == "orange")
  {
    LogicalVolume->SetVisAttributes(orange);
  }
  else if (Color == "yellow")
  {
    LogicalVolume->SetVisAttributes(yellow);
  }
  else if (Color == "green")
  {
    LogicalVolume->SetVisAttributes(green);
  }
  else if (Color == "cyan")
  {
    LogicalVolume->SetVisAttributes(cyan);
  }
  else if (Color == "blue")
  {
    LogicalVolume->SetVisAttributes(blue);
  }
  else if (Color == "magenta")
  {
    LogicalVolume->SetVisAttributes(magenta);
  }
}

void BREMCANNONSimGeometry::SetLogicalVolumeColorFromMaterial(G4LogicalVolume *LogicalVolume, G4Material *Material)
{

  auto Alu = G4NistManager::Instance()->FindOrBuildMaterial("G4_Al");
  auto Cuivre = G4NistManager::Instance()->FindOrBuildMaterial("G4_Cu");
  auto Etain = G4NistManager::Instance()->FindOrBuildMaterial("G4_Sn");
  auto Plomb = G4NistManager::Instance()->FindOrBuildMaterial("G4_Pb");
  auto Tantale = G4NistManager::Instance()->FindOrBuildMaterial("G4_Ta");

  // ***********************
  // Visualization Colors
  // ***********************
  // Create some colors for visualizations
  invis = new G4VisAttributes(G4Colour(255 / 255., 255 / 255., 255 / 255.));
  invis->SetVisibility(false);

  white = new G4VisAttributes(G4Colour(1, 1, 1, 0.1)); // Sets the color (can be looked up online)
  // white->SetForceWireframe(true); // Sets to wire frame mode for coloring the volume
  white->SetForceSolid(true); // Sets to solid mode for coloring the volume
  white->SetVisibility(true); // Makes color visible in visualization

  gray = new G4VisAttributes(G4Colour(0.5, 0.5, 0.5, 1));
  //  gray->SetForceWireframe(true);
  gray->SetForceSolid(true);
  gray->SetVisibility(true);

  black = new G4VisAttributes(G4Colour(0, 0, 0, 1));
  //  black->SetForceWireframe(true);
  black->SetForceSolid(true);
  black->SetVisibility(true);

  red = new G4VisAttributes(G4Colour(1, 0, 0, 1));
  //  red->SetForceWireframe(true);
  red->SetForceSolid(true);
  red->SetVisibility(true);

  orange = new G4VisAttributes(G4Colour(1, 0.5, 0, 1));
  //  orange->SetForceWireframe(true);
  orange->SetForceSolid(true);
  orange->SetVisibility(true);

  yellow = new G4VisAttributes(G4Colour(1, 1, 0, 0.29));
  //  yellow->SetForceWireframe(true);
  yellow->SetForceSolid(true);
  yellow->SetVisibility(true);

  green = new G4VisAttributes(G4Colour(0, 1, 0, 0.35));
  //  green->SetForceWireframe(true);
  green->SetForceSolid(true);
  green->SetVisibility(true);

  cyan = new G4VisAttributes(G4Colour(0, 1, 1, 0.55));
  //  cyan->SetForceWireframe(true);
  cyan->SetForceSolid(true);
  cyan->SetVisibility(true);

  blue = new G4VisAttributes(G4Colour(0, 0, 1, 1));
  //  blue->SetForceWireframe(true);
  blue->SetForceSolid(true);
  blue->SetVisibility(true);

  magenta = new G4VisAttributes(G4Colour(1, 0, 1, 0.85));
  //  magenta->SetForceWireframe(true);
  // magenta->SetForceSolid(true);
  magenta->SetVisibility(true);

  if (Material == Plomb)
  {
    LogicalVolume->SetVisAttributes(black);
  }
  // else if (Material == "white")
  // {
  //   LogicalVolume->SetVisAttributes(white);
  // }
  else if (Material == Alu)
  {
    LogicalVolume->SetVisAttributes(gray);
  }
  // else if (Material == "red")
  // {
  //   LogicalVolume->SetVisAttributes(red);
  // }
  else if (Material == Cuivre)
  {
    LogicalVolume->SetVisAttributes(orange);
  }
  else if (Material == Tantale)
  {
    LogicalVolume->SetVisAttributes(yellow);
  }
  // else if (Material == "green")
  // {
  //   LogicalVolume->SetVisAttributes(green);
  // }
  // else if (Material == "cyan")
  // {
  //   LogicalVolume->SetVisAttributes(cyan);
  // }
  else if (Material == Etain)
  {
    LogicalVolume->SetVisAttributes(blue);
  }
  // else if (Material == "magenta")
  // {
  //   LogicalVolume->SetVisAttributes(magenta);
  // }
}

void BREMCANNONSimGeometry::ConstructWorld()
{
  G4RotationMatrix DontRotate;
  DontRotate.rotateZ(0 * deg);
  DontRotate.rotateX(0 * deg);

  auto Material = G4NistManager::Instance()->FindOrBuildMaterial("G4_Galactic");

  auto SolidWorld = new G4Box("SolidWorld", 110 * cm, 110 * cm, 110 * cm);
  LogicalWorld = new G4LogicalVolume(SolidWorld, Material, "LogicalWorld", 0, 0, 0);

  // Place the world volume: center of world at origin (0,0,0)
  PhysicalWorld = new G4PVPlacement(G4Transform3D(DontRotate, G4ThreeVector(0, 0, 0)), "PhysicalWorld", LogicalWorld, NULL, false, 0);
}

void BREMCANNONSimGeometry::ConstructMaterialsList()
{
  for (int i = 0; i < Number_IP - 1; i++)
  {
    //G4cout << "MaterialsName[" << i << "] = " << Geom->GetMaterialFilter(i) << G4endl;
    if (Geom->GetMaterialFilter(i) == "Alu")
      Material_Filter[i] = Alu;
    else if (Geom->GetMaterialFilter(i) == "Cuivre")
      Material_Filter[i] = Cuivre;
    else if (Geom->GetMaterialFilter(i) == "Etain")
      Material_Filter[i] = Etain;
    else if (Geom->GetMaterialFilter(i) == "Plomb")
      Material_Filter[i] = Plomb;
    else if (Geom->GetMaterialFilter(i) == "Titane")
      Material_Filter[i] = Titane;
    else if (Geom->GetMaterialFilter(i) == "Fer")
      Material_Filter[i] = Fer;
    else if (Geom->GetMaterialFilter(i) == "Tantale")
      Material_Filter[i] = Tantale;
    else
    {
      G4Exception("Materials", "materials0001", FatalException,
                  "Problem with MaterialsList Construction");
    }
    //G4cout << "Materials[" << i << "] = " << Material_Filter[i] << G4endl;
  }
}

void BREMCANNONSimGeometry::ConstructFiltersAndIPs()
{
  G4RotationMatrix *stack_rot = new G4RotationMatrix;
  // G4double theta = -12*deg;
  stack_rot->rotateZ(0 * deg);
  stack_rot->rotateX(0 * deg);

  auto Sensitive = BREMCANNONSimMaterials::getInstance()->GetMaterial("Sensitive");
  auto Base = BREMCANNONSimMaterials::getInstance()->GetMaterial("Base");
  auto Polyester = BREMCANNONSimMaterials::getInstance()->GetMaterial("Polyester");
  Alu = G4NistManager::Instance()->FindOrBuildMaterial("G4_Al");
  Cuivre = G4NistManager::Instance()->FindOrBuildMaterial("G4_Cu");
  Etain = G4NistManager::Instance()->FindOrBuildMaterial("G4_Sn");
  Plomb = G4NistManager::Instance()->FindOrBuildMaterial("G4_Pb");
  Titane = G4NistManager::Instance()->FindOrBuildMaterial("G4_Ti");
  Fer = G4NistManager::Instance()->FindOrBuildMaterial("G4_Fe");
  Tantale = G4NistManager::Instance()->FindOrBuildMaterial("G4_Ta");

  ConstructMaterialsList();

  auto LogicalStackIP1 = Geom->GetStackIP1("Stack_IP1", Polyester);
  auto LogicalStackIP2 = Geom->GetStackIP2("Stack_IP2", Sensitive);
  auto LogicalStackIP3 = Geom->GetStackIP3("Stack_IP3", Polyester);
  auto LogicalStackIP4 = Geom->GetStackIP4("Stack_IP4", Base);

  // SetLogicalVolumeColor(LogicalFiltre14_15, "black");
  SetLogicalVolumeColor(LogicalStackIP1, "gray");
  SetLogicalVolumeColor(LogicalStackIP2, "red");
  SetLogicalVolumeColor(LogicalStackIP3, "green");
  SetLogicalVolumeColor(LogicalStackIP4, "black");

  Zposition = Stack_IP_pos_ini + Disk_thickness;

  if (flag_Mylar == 1)
    Zposition += Mylar_thickness;

  for (int i = 0; i < Number_IP; i++)
  {
    // for (int i=0;i<1;i++) {
    G4String number = std::to_string(i);
    G4String name;

    Zposition += IPa1_z / 2.;

    // Création des volumes Stack_IP
    G4LogicalVolume *logicalVolumes[] = {LogicalStackIP1, LogicalStackIP2, LogicalStackIP3, LogicalStackIP4};
    G4double thicknesses[] = {IPa1_z, IPa2_z, IPa3_z, IPa4_z};

    for (int j = 0; j < 4; j++)
    {
      name = "Stack_IP" + std::to_string(j + 1) + "_" + number;
      physiStackIP1[i] = new G4PVPlacement(stack_rot,                               // no rotation
                                           G4ThreeVector(0, 0, Zposition), // position
                                           logicalVolumes[j],                       // its logical volume
                                           name,                                     // its name
                                           LogicalWorld,                            // its mother volume
                                           false,                                   // no boolean operations
                                           0,
                                           false);
      Zposition += thicknesses[j] / 2.;
      if (j < 3)
      {
        Zposition += thicknesses[j + 1] / 2.;
      }
      if (j==0) PosZSensitive[i] = Zposition - thicknesses[j+1]/2;
    }

    name = "disk_" + number;
    G4double thickness_Filter = 0.;

    if (i < Number_IP - 1)
    {
      thickness_Filter = Geom->GetThicknessFilter(i);
      Zposition += thickness_Filter / 2.;
      LogicalFilter[i] = Geom->GetFilter(name, Radius_IP, Geom->GetThicknessFilter(i), Material_Filter[i]);
      SetLogicalVolumeColorFromMaterial(LogicalFilter[i], Material_Filter[i]);
      // G4cout << "Material[" << i << "] = " << Material_filter[i] << G4endl;
      physiDisk[i] = new G4PVPlacement(stack_rot,                               // no rotation
                                         G4ThreeVector(0, 0, Zposition), // position
                                         LogicalFilter[i],                        // its logical volume
                                         name,                                     // its name
                                         LogicalWorld,                            // its mother  volume
                                         false,                                   // no boolean operations
                                         0,
                                         false);
      Zposition += thickness_Filter / 2.;
    }
  }

  G4double distance_free = Cylinder_tantale_length - Zposition + Disk_thickness - 0.1;
  if (distance_free <= 0)
  {
    G4Exception("Geometry", "geometry0001", FatalException,
                "Thickness of all IPs & Filters > Length Cylinder");
  }
  Zposition = Zposition + distance_free / 2;

  auto LogicalCylinderLead = Geom->GetVolumeFiller("Cylinder_Lead", Plomb, distance_free);
  SetLogicalVolumeColor(LogicalCylinderLead, "black");

  auto Cylinder_Lead_phys = new G4PVPlacement(stack_rot,                               // no rotation
                                               G4ThreeVector(0, 0, Zposition), // position
                                               LogicalCylinderLead,                    // its logical volume
                                               "Cylinder_Lead",
                                               LogicalWorld, // its mother  volume
                                               false,        // no boolean operations
                                               0,
                                               false);
}

void BREMCANNONSimGeometry::ConstructCylinderPart(G4int flag)
{
  G4RotationMatrix *stack_rot = new G4RotationMatrix;
  // G4double theta = -12*deg;
  stack_rot->rotateZ(0 * deg);
  stack_rot->rotateX(0 * deg);

  auto Alu = G4NistManager::Instance()->FindOrBuildMaterial("G4_Al");
  auto Tantale = G4NistManager::Instance()->FindOrBuildMaterial("G4_Ta");
  auto Cuivre = G4NistManager::Instance()->FindOrBuildMaterial("G4_Cu");
  auto Plomb = G4NistManager::Instance()->FindOrBuildMaterial("G4_Pb");

  auto LogicalFullDisk = Geom->GetFullDisk("FullDisk", Alu);
  auto LogicalOpenDisk = Geom->GetOpenDisk("OpenDisk", Alu);
  auto LogicalEntryProtection = Geom->GetEntryProtection("Mylar_Entree", Alu);
  auto LogicalCorpus = Geom->GetCorpus("Corpus_Tantale", Tantale);

  // Set colors of various block materials
  SetLogicalVolumeColor(LogicalWorld, "invis");
  SetLogicalVolumeColor(LogicalFullDisk, "gray");
  SetLogicalVolumeColor(LogicalOpenDisk, "gray");
  SetLogicalVolumeColor(LogicalEntryProtection, "gray");
  SetLogicalVolumeColor(LogicalCorpus, "yellow");

  Zposition = Stack_IP_pos_ini + Disk_thickness / 2;
  if (flag == 1)
  {
    auto Cylinder_Disk_entry = new G4PVPlacement(stack_rot,                               // no rotation
                                                    G4ThreeVector(0, 0, Zposition), // position
                                                    LogicalOpenDisk,                         // its logical volume
                                                    "OpenDisk_Entry",
                                                    LogicalWorld, // its mother  volume
                                                    false,        // no boolean operations
                                                    0,
                                                    false);

    Zposition += Disk_thickness / 2 + Mylar_thickness / 2;

    auto PhysicalEntryProtection = new G4PVPlacement(stack_rot,                               // no rotation
                                                  G4ThreeVector(0, 0, Zposition), // position
                                                  LogicalEntryProtection,                  // its logical volume
                                                  "EntryProtection",
                                                  LogicalWorld, // its mother  volume
                                                  false,        // no boolean operations
                                                  0,
                                                  false);
  }
  else
  {
    auto Cylinder_Disk_entry = new G4PVPlacement(stack_rot,                               // no rotation
                                                    G4ThreeVector(0, 0, Zposition), // position
                                                    LogicalFullDisk,                         // its logical volume
                                                    "FullDisk_Entry",
                                                    LogicalWorld, // its mother  volume
                                                    false,        // no boolean operations
                                                    0,
                                                    false);
  }

  Zposition = Stack_IP_pos_ini + Disk_thickness + Cylinder_tantale_length / 2.;

  auto Cylinder_Tantale_phys = new G4PVPlacement(stack_rot,                               // no rotation
                                                 G4ThreeVector(0, 0, Zposition), // position
                                                 LogicalCorpus,                           // its logical volume
                                                 "Corpus_Tantale",
                                                 LogicalWorld, // its mother  volume
                                                 false,        // no boolean operations
                                                 0,
                                                 false);

  Zposition = Stack_IP_pos_ini + Disk_thickness + Cylinder_tantale_length + Disk_thickness / 2;

  auto Cylinder_Disk_exit = new G4PVPlacement(stack_rot,                               // no rotation
                                                  G4ThreeVector(0, 0, Zposition), // position
                                                  LogicalFullDisk,                         // its logical volume
                                                  "FullDisk Exit",
                                                  LogicalWorld, // its mother  volume
                                                  false,        // no boolean operations
                                                  0,
                                                  false);
}

void BREMCANNONSimGeometry::GetVariables()
{
  // Initialize scint classes
  Geom = new Geometry(path_bin + "BREMCANNONSim.cfg");

  // ***********************
  // Various dimensions
  // ***********************
  Radius_cylinder_external = Geom->GetRadiusCylinderExternal();
  Radius_cylinder_internal = Geom->GetRadiusCylinderInternal();
  Disk_thickness = Geom->GetDiskThickness();
  Mylar_thickness = Geom->GetMylarThickness();
  flag_Mylar = Geom->GetFlagMylar();
  Cylinder_tantale_length = Geom->GetCylinderTantaleLength();
  IPa1_z = Geom->GetIPa1z();
  IPa2_z = Geom->GetIPa2z();
  IPa3_z = Geom->GetIPa3z();
  IPa4_z = Geom->GetIPa4z();
  Radius_IP = Geom->GetRadiusIP();
  Thickness_Filter[1] = Geom->GetThicknessFilter(1);
  Thickness_Filter[2] = Geom->GetThicknessFilter(2);
  Thickness_Filter[3] = Geom->GetThicknessFilter(3);
  Thickness_Filter[4] = Geom->GetThicknessFilter(4);
  Thickness_Filter[5] = Geom->GetThicknessFilter(5);
  Thickness_Filter[6] = Geom->GetThicknessFilter(6);
  Thickness_Filter[7] = Geom->GetThicknessFilter(7);
  Thickness_Filter[8] = Geom->GetThicknessFilter(8);
  Thickness_Filter[9] = Geom->GetThicknessFilter(9);
  Thickness_Filter[10] = Geom->GetThicknessFilter(10);
  Thickness_Filter[11] = Geom->GetThicknessFilter(11);
  Thickness_Filter[12] = Geom->GetThicknessFilter(12);
  Thickness_Filter[13] = Geom->GetThicknessFilter(13);
  Thickness_Filter[14] = Geom->GetThicknessFilter(14);
  Labs = Geom->GetLabs();
  Coef_PSL = Geom->GetCoefPSL();
  G4double thickness_IP = IPa1_z + IPa2_z + IPa3_z + IPa4_z;
  G4double Zposition = 0;
  Number_IP = Geom->GetNumberIP();
}

// Main Function: Builds Full block, coupling, and PMT geometries
G4VPhysicalVolume *BREMCANNONSimGeometry::Construct()
{
  GetVariables();

  ConstructWorld();
  ConstructCylinderPart(flag_Mylar);
  ConstructFiltersAndIPs();

  // Returns world with everything in it and all properties set
  return PhysicalWorld;
}
