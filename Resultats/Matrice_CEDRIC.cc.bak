#include "Matrice_CEDRIC.hh"

void Matrice_CEDRIC()
{

  file = new TFile("CEDRIC_Matrice_LULI.root");
  Matrice = (TH2F*)file->Get("matrice");
  Matrice->Draw("colz");

  new TCanvas;
  Create_Histo_Simulated(100, 20, 60);
  Fit->Draw();
  
  //################################################################################
  //################################################################################
  //##############################PARTIE FIT MINUIT#################################
  //################################################################################
  //################################################################################
  
  Emin =1;
  Emax =100;
  Nsimulated = 1e8;

  
  FOM_E0 = FOM_Spectro_FIT(Emin, Emax);
  FOM_N = Integral_Spectro_FIT(Emin, Emax);

  Draw_Results_FOM();

  Create_Pad_Canvas("RESULTS GAMMA");
  pad1->cd();
  pad1->SetLogy();
  //REF = Histo_PSL("Exp_10_20_60_10M.root");
  REF = Histo_PSL("Exp_30keV_100M.root");
  REF->Draw();
  FIT_MINUIT();
  TH1F* h_1Temp = (TH1F*)PSL_Rec->Clone("h_1Temp");
  REF->Draw();
  h_1Temp->Draw("same");
  FIT_MINUIT_2Temp();
  TH1F* h_2Temp = (TH1F*)PSL_Rec->Clone("h_2Temp");
  h_2Temp->Draw("same");
  
  pad2->cd();
  pad2->SetLogy();
  Draw_Incident_Spectrum_Fit(10, 30, 60, 1);
  
}
