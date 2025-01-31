#include <fstream>
TH1D* PSL_REF;
TH1D* Edep_REF;
TH1D* h;
TFile* file;
TH2F* Matrice;
TH1F* IncidentE = new TH1F("IncidentE", "IncidentE", 100, 0, 100);
TH1D* PSL_Rec = new TH1D ("PSL_Rec", "PSL_Rec", 15, 0.5, 15.5);
TH1F *Fit = new TH1F("fit", "fit", 1000, 0, 1000);
TH1D* REF;
float Chi2=0;
float Chi21T=0;
float Chi22T=0;
double A=0;
double err_A=0;
double E0=0;
double err_E0=0;
double Acold=0;
double err_Acold=0;
double Ecold=0;
double err_Ecold=0;
double Ahot=0;
double err_Ahot=0;
double Ehot=0;
double err_Ehot=0;
double NEvents=0;
double Nsimulated=0;
float Emin=0;
float Emax=0;
TGraph* Table_Chi2_995pc;
TGraph* Table_Chi2_990pc;
TGraph* Table_Chi2_975pc;
TGraph* Table_Chi2_950pc;
TGraph* Table_Chi2_900pc;
TGraph* Table_Chi2_750pc;
TGraph* Table_Chi2_500pc;
TGraph* Table_Chi2_250pc;
TGraph* Table_Chi2_100pc;
TGraphErrors *FOM_E0;
TGraphErrors *FOM_N;
TF1* f1;
double N=0;
double eN=0;
float solid_angle=0;
TPad *pad1;
TPad *pad2;
TCanvas* c;

float start_norm = 1e8;
float step_norm = 1e7;
float low_norm = 1e1;
float up_norm = 1e12;

float start_E = 40;//10
float step_E = 0.1;
float low_E = 0.1; //-1
float up_E = 200; //30

float start_Ecold = 10;
float step_Ecold = 1;
float low_Ecold = 1;
float up_Ecold = 30; 

float start_Ehot = 50;
float step_Ehot = 1;
float low_Ehot = 20; 
float up_Ehot = 150; 


void Init_Table_Chi2()
{
  TFile* file = new TFile("Table_Chi2_995.root");
  Table_Chi2_995pc = (TGraph*)file->Get("Graph");

  file = new TFile("Table_Chi2_990.root");
  Table_Chi2_990pc = (TGraph*)file->Get("Graph");

  file = new TFile("Table_Chi2_975.root");
  Table_Chi2_975pc = (TGraph*)file->Get("Graph");

  file = new TFile("Table_Chi2_950.root");
  Table_Chi2_950pc = (TGraph*)file->Get("Graph");

  file = new TFile("Table_Chi2_900.root");
  Table_Chi2_900pc = (TGraph*)file->Get("Graph");

  file = new TFile("Table_Chi2_750.root");
  Table_Chi2_750pc = (TGraph*)file->Get("Graph");

  file = new TFile("Table_Chi2_500.root");
  Table_Chi2_500pc = (TGraph*)file->Get("Graph");

  file = new TFile("Table_Chi2_250.root");
  Table_Chi2_250pc = (TGraph*)file->Get("Graph");

  file = new TFile("Table_Chi2_100.root");
  Table_Chi2_100pc = (TGraph*)file->Get("Graph");

}


void Create_Pad_Canvas(const char* name)
{
  gStyle->SetOptTitle(kFALSE);
  c = new TCanvas(name, name, 0, 10, 2500, 1000);
  c->SetWindowPosition(600, 0);
  pad1 = new TPad("pad1", "", 0.0, 0.0, 0.5, 1);
  pad2 = new TPad("pad2", "", 0.51, 0.0, 1, 1);

  pad1->Draw();
  pad2->Draw();

}


double NumberOfEntries(const char* filename)
{
  TFile* file = new TFile(filename);
  TTree* Tree = (TTree*)file->Get("IP");
  int Entries = Tree->GetEntries();

  file->Close();

  return Entries;
}



TH1D* Histo_PSL(const char* filename)
{
  file = new TFile(filename, "update");
  //PSL_REF = (TH1D*)file->Get("PSL vs IP");
  Edep_REF = (TH1D*)file->Get("edep_totale_eff");
  //Edep_REF = (TH1D*)file->Get("edep_totale");
  //Edep_REF->Scale(0.001);
  //PSL_REF->Scale(1/(3.1415927*7*7));

  int n = Edep_REF->GetNbinsX();
  NEvents = NumberOfEntries(filename);
  double Integral = Edep_REF->Integral();
  double Error_Edep=0;
  double Error_PSL=0;
  double Coef_PSL = 6.95E-1;

   for (int i=1; i<=n; i++)
     {
       Error_Edep = sqrt((Edep_REF->Integral(i,i)*Integral)/NEvents); //si graph Edep
       //Error_PSL = (Error_Edep*Coef_PSL)/(3.1415927*7*7); // si graph PSL
       Edep_REF->SetBinError(i, Error_Edep);
       //PSL_REF->SetBinError(i, Error_PSL);
     }


   //PSL_REF->SetDirectory(nullptr);
  Edep_REF->SetDirectory(nullptr);
  file->Close();
  //delete file;

  //return PSL_REF;
  return Edep_REF;

}


bool Test_Fichier(const char* filename)
{

  bool fichier=true;
  file = TFile::Open(filename);
  if(!file)
    {
      fichier=false;
      //cout << "Fichier [" << filename << "] non présent" << endl;
    }
  
  return fichier;
}
  



void Calcul_Chi2(TH1D* O, TH1D* C)
{
  Chi2=0;
  int n = C->GetNbinsX();
  //  cout << "n = " << n << endl;

  //O->Scale(C->Integral()/O->Integral());

  float sigma=0;

  for (int i=1; i<=n; i++)
    {
      sigma = C->GetBinError(i);
      //Chi2+= ((O->GetBinContent(i) - C->GetBinContent(i)) * (O->GetBinContent(i) - C->GetBinContent(i)) / (C->GetBinContent(i)));
      if(C->GetBinContent(i) !=0 && sigma !=0)
	{
	  Chi2+= ((O->GetBinContent(i) - C->GetBinContent(i)) * (O->GetBinContent(i) - C->GetBinContent(i))) / (C->GetBinContent(i) + (sigma*sigma));

	}
    }

}




void Histo_Exp(float E)
{
  TF1* f1 = new TF1("f1", "exp(-x/[0])", 0, 1000);
  f1->SetParameter(0, E);
  Fit->Reset();

  for(int i=0; i<1000000; i++)
    {
      Fit->Fill(f1->GetRandom());
    }


}



void Create_Histo_Simulated(float norm_Ecold, float Ecold, float Ehot)
{
  TF1* f1 = new TF1("f1", "[0]*exp(-x/[1]) + exp(-x/[2])", 0, 1000);
  f1->SetParameter(0, norm_Ecold);
  f1->SetParameter(1, Ecold);
  f1->SetParameter(2, Ehot);
  Fit->Reset();

  for(int i=0; i<10000000; i++)
    {
      Fit->Fill(f1->GetRandom());
    }
  Fit->Scale(1./10000000);


  ofstream output ("../bin/Data_Exp_100_20_60.txt");

  for (float i=1; i<=1000; i++)
    {
      output << i/1000 << " " << Fit->GetBinContent(i) << endl;
    }

  output.close();

}



void Reconstruction_PSL()
{
  PSL_Rec->Reset();
  
  PSL_Rec->SetLineColor(kRed);

  for (int i=1; i<100; i++)
    {
      for (int j=1; j<=15; j++)
	{
	  PSL_Rec->SetBinContent(j, PSL_Rec->GetBinContent(j) + Fit->GetBinContent(i)*Matrice->GetBinContent(j, i));
	}
    }

}





Double_t fitFunc(float x, Double_t* par)
{
  Double_t PDF =0.0;
  PDF = 10*par[0]*exp(-(x*10)/par[1]);
  // cout << "x = " << x << endl;
  // cout << "par 0 =" << par[0] << endl;
  // cout << "par 1 =" << par[1] << endl;
  // cout << "PDF = " << PDF << endl;

  return PDF;

}





void fcn(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag)
{
  PSL_Rec->Reset();
  
  for (int i=1; i<=100; i++)
    {
      for (int j=1; j<=15; j++)
	{
	  PSL_Rec->SetBinContent(j, PSL_Rec->GetBinContent(j) + fitFunc(i-0.5, par)*Matrice->GetBinContent(j, i));
	}
    }



  Calcul_Chi2(PSL_Rec, REF);
  f = Chi2;
  //cout << "A = " << par[0] << endl;
  //cout << "E0 = " << par[1] << endl;
  //cout << "Chi2 = " << Chi2 << endl;

}




void FIT_MINUIT()
{
  TMinuit *gMinuit = new TMinuit(2);  //initialize TMinuit with a maximum of 4 params
  gMinuit->SetFCN(fcn);

  Double_t arglist[10];
  Int_t ierflg = 0;
  
  arglist[0] = 1;
  gMinuit->mnexcm("SET ERR", arglist ,1,ierflg);
  
  // Set starting values and step sizes for parameters
  static Double_t vstart[2] = {100, 1};
  static Double_t step[2] = {10.0 , 0.1};
  gMinuit->mnparm(0, "A", start_norm, step_norm, low_norm, up_norm, ierflg);
  gMinuit->mnparm(1, "E0", start_E, step_E, low_E, up_E, ierflg);
  
  // Now ready for minimization step
  arglist[0] = 500;
  arglist[1] = 1.;

  //  gMinuit->mnexcm("CALL FCN", arglist ,1,ierflg);
  gMinuit->mnexcm("MIGRAD", arglist ,2,ierflg);
  
  // Print results
  Double_t amin,edm,errdef;
  Int_t nvpar,nparx,icstat;
  gMinuit->mnstat(amin,edm,errdef,nvpar,nparx,icstat);


  //new TCanvas;
  //PSL_Rec->Draw();
  //REF->Draw("same");
  REF->SetLineColor(kBlack);
  PSL_Rec->SetLineColor(kRed);

  Chi21T = Chi2;

  gMinuit->GetParameter(0, A, err_A);
  gMinuit->GetParameter(1, E0, err_E0);
  cout << "A = " << A << " +- " << err_A << endl;
  cout << "E0 = " << E0 << " +- " << err_E0 << endl;
  cout << "N = " << (A*E0) << " +- " << (E0*err_A + A*err_E0) << endl;
  cout << "Chi2 = " << Chi21T << endl;
  cout << "Fit status = " << gMinuit->GetStatus() << endl;
}






Double_t fitFunc2T(float x, Double_t* par)
{
  Double_t PDF =0.0;
  PDF = 10*par[0]*exp(-(x*10)/par[1]) +10*par[2]*exp(-(x*10)/par[3]);
  // cout << "x = " << x << endl;
  // cout << "par 0 =" << par[0] << endl;
  // cout << "par 1 =" << par[1] << endl;
  // cout << "PDF = " << PDF << endl;

  return PDF;

}


void fcn2T(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag)
{
  PSL_Rec->Reset();
  
  for (int i=1; i<=100; i++)
    {
      for (int j=1; j<=15; j++)
	{
	  PSL_Rec->SetBinContent(j, PSL_Rec->GetBinContent(j) + fitFunc2T(i-0.5, par)*Matrice->GetBinContent(j, i));
	}
    }



  Calcul_Chi2(PSL_Rec, REF);
  f = Chi2;
  //cout << "A = " << par[0] << endl;
  //cout << "E0 = " << par[1] << endl;
  //cout << "Chi2 = " << Chi2 << endl;

}



void FIT_MINUIT_2Temp()
{
  TMinuit *gMinuit = new TMinuit(4);  //initialize TMinuit with a maximum of 4 params
  gMinuit->SetFCN(fcn2T);

  Double_t arglist[10];
  Int_t ierflg = 0;
  
  arglist[0] = 1;
  gMinuit->mnexcm("SET ERR", arglist ,1,ierflg);
  
  // Set starting values and step sizes for parameters
  gMinuit->mnparm(0, "Acold", start_norm, step_norm, low_norm, up_norm, ierflg);
  gMinuit->mnparm(1, "Ecold", start_Ecold, step_Ecold, low_Ecold, up_Ecold, ierflg);
  gMinuit->mnparm(2, "Ahot", start_norm, step_norm, low_norm, up_norm, ierflg);
  gMinuit->mnparm(3, "Ehot", start_Ehot, step_Ehot, low_Ehot, up_Ehot, ierflg);
  
  // Now ready for minimization step
  arglist[0] = 1000;
  arglist[1] = 1.;

  //  gMinuit->mnexcm("CALL FCN", arglist ,1,ierflg);
  gMinuit->mnexcm("MIGRAD", arglist ,4,ierflg);
  
  // Print results
  Double_t amin,edm,errdef;
  Int_t nvpar,nparx,icstat;
  gMinuit->mnstat(amin,edm,errdef,nvpar,nparx,icstat);


  PSL_Rec->SetLineColor(kBlue);
  Chi22T = Chi2;

  gMinuit->GetParameter(0, Acold, err_Acold);
  gMinuit->GetParameter(1, Ecold, err_Ecold);
  gMinuit->GetParameter(2, Ahot, err_Ahot);
  gMinuit->GetParameter(3, Ehot, err_Ehot);
  cout << "Acold = " << Acold << " +- " << err_Acold << endl;
  cout << "Ecold = " << Ecold << " +- " << err_Ecold << endl;
  cout << "Ahot = " << Ahot << " +- " << err_Ahot << endl;
  cout << "Ehot = " << Ehot << " +- " << err_Ehot << endl;
  cout << "N = " << (Acold*Ecold+Ahot*Ehot) << " +- " << (Ecold*err_Acold + Acold*err_Ecold + Ehot*err_Ahot + Ahot*err_Ehot) << endl;
  cout << "Chi2 = " << Chi22T << endl;
  cout << "Fit status = " << gMinuit->GetStatus() << endl;
}





TGraphErrors* FOM_Spectro_FIT(float Emin, float Emax)
{
  char filename[100];
  int npoints = ((Emax - Emin))+1;
  float x[npoints];
  float ex[npoints];
  float y[npoints];
  float ey[npoints];
  float num = Emin;
  cout << "npts = " << npoints << endl;


    for (int i=0; i<npoints; i++)
    {
      sprintf(filename,"Exp_%dkeV_100M.root", (int)num);
      cout << "filename = " << filename << endl;
      REF = Histo_PSL(filename);
      FIT_MINUIT();
      x[i] = num;
      ex[i] = 0.5;
      y[i] = E0;
      ey[i] = err_E0;
      
      num+=1;
     }

    auto ga = new TGraphErrors(npoints, x, y, ex, ey);

    return ga;
  
}



TGraphErrors* Integral_Spectro_FIT(float Emin, float Emax)
{
  char filename[100];
  int npoints = ((Emax - Emin))+1;
  float x[npoints];
  float ex[npoints];
  float y[npoints];
  float ey[npoints];
  float num = Emin;
  cout << "npts = " << npoints << endl;

  f1 = new TF1("f1", "[0]*exp(-x/[1])", 0, 1000);


    for (int i=0; i<npoints; i++)
    {
      sprintf(filename,"Exp_%dkeV_100M.root", (int)num);
      cout << "filename = " << filename << endl;
      REF = Histo_PSL(filename);
      FIT_MINUIT();
      f1->SetParameter(0, A);
      f1->SetParameter(1, E0);
      x[i] = num;
      ex[i] = 0.5;
      y[i] = f1->Integral(0, 1000);
      ey[i] = sqrt(f1->Integral(0, 1000));
      
      num+=1;
     }

    auto ga = new TGraphErrors(npoints, x, y, ex, ey);

    return ga;
  
}



void Draw_Results_FOM()
{
  TCanvas *canvas1 = new TCanvas("FOM", "FOM", 0, 10, 2500, 1500);
  TPad *p1;
  TPad *p2;
  TPad *p3;
  TPad *p4;
  p1 = new TPad("p1", "", 0, 0.34, 0.49, 1);
  p2 = new TPad("p2", "", 0., 0., 0.49, 0.33);
  p3 = new TPad("p3", "", 0.51, 0.34, 1, 1);
  p4 = new TPad("p4", "", 0.51, 0., 1, 0.33);
  p1->Draw();
  p2->Draw();
  p3->Draw();
  p4->Draw();
  
  p1->cd();
  p1->SetGridx();
  p1->SetGridy();
  float x[2] = {0, 100};
  float y[2] = {0, 100};
  auto line = new TGraph(2, x , y);
  FOM_E0->SetFillColor(kRed);
  FOM_E0->SetLineColor(kRed);
  FOM_E0->SetLineWidth(3);
  FOM_E0->SetFillStyle(3001);
  FOM_E0->Draw("a3PL");
  FOM_E0->GetXaxis()->SetTitle("True Energy k_{B}T_{e} [keV]");
  FOM_E0->GetYaxis()->SetTitle("Reconstructed Energy k_{B}T_{e} [keV]");
  line->Draw("Lsame");
  line->SetLineColor(kCyan);
  line->SetLineWidth(3);

  
  p2->cd();
  p2->SetGridx();
  p2->SetGridy();
  int n = FOM_E0->GetN();
  float ecart_x[n];
  float ecart_y[n];
  float e_ecart_x[n];
  float e_ecart_y[n];


  for (int i =0; i<n; i++)
    {
      ecart_x[i] = FOM_E0->GetPointX(i);
      ecart_y[i] = 100*((FOM_E0->GetPointY(i) - FOM_E0->GetPointX(i) ) / FOM_E0->GetPointX(i));
      e_ecart_x[i] = 0.5;
      e_ecart_y[i] = 100*(FOM_E0->GetErrorY(i) / FOM_E0->GetPointX(i));
      
    }

  auto ega = new TGraphErrors(n, ecart_x, ecart_y, e_ecart_x, e_ecart_y);
  ega->Draw("a3PL");
  ega->SetFillColor(kRed);
  ega->SetLineColor(kRed);
  ega->SetLineWidth(3);
  ega->SetFillStyle(3001);
  ega->GetXaxis()->SetTitle("True Energy k_{B}T_{e} [keV]");
  ega->GetYaxis()->SetTitle("#frac{Fit-Ref}{Ref}");
  float xx[2] = {0, 100};
  float yy[2] = {0, 0};
  auto line1 = new TGraph(2, xx , yy);
  line1->Draw("Lsame");
  line1->SetLineColor(kCyan);
  line1->SetLineWidth(3);


  p3->cd();
  p3->SetGridx();
  p3->SetGridy();

  float x2[2] = {0, 100};
  float y2[2] = {1e8, 1e8};
  auto line2 = new TGraph(2, x2 , y2);
  FOM_N->SetFillColor(kRed);
  FOM_N->SetLineColor(kRed);
  FOM_N->SetLineWidth(3);
  FOM_N->SetFillStyle(3001);
  FOM_N->Draw("a3PL");
  FOM_N->GetYaxis()->SetRangeUser(8e7, 1.2e8);
  FOM_N->GetXaxis()->SetTitle("True Energy k_{B}T_{e} [keV]");
  FOM_N->GetYaxis()->SetTitle("Reconstructed N_{events}");
  line2->Draw("Lsame");
  line2->SetLineColor(kCyan);
  line2->SetLineWidth(3);

  
  p4->cd();
  p4->SetGridx();
  p4->SetGridy();
  int m = FOM_N->GetN();
  float ecart_x2[m];
  float ecart_y2[m];
  float e_ecart_x2[m];
  float e_ecart_y2[m];


  for (int i =0; i<m; i++)
    {
      ecart_x2[i] = FOM_N->GetPointX(i);
      ecart_y2[i] = 100*((FOM_N->GetPointY(i) - Nsimulated ) / Nsimulated);
      e_ecart_x2[i] = 0.5;
      e_ecart_y2[i] = 100*((FOM_N->GetErrorY(i)/Nsimulated) + (FOM_N->GetPointY(i)*sqrt(Nsimulated))/(Nsimulated*Nsimulated));
      
    }

  auto ega2 = new TGraphErrors(n, ecart_x2, ecart_y2, e_ecart_x2, e_ecart_y2);
  ega2->Draw("a3PL");
  ega2->SetFillColor(kRed);
  ega2->SetLineColor(kRed);
  ega2->SetLineWidth(3);
  ega2->SetFillStyle(3001);
  ega2->GetYaxis()->SetRangeUser(-20, 20);
  ega2->GetXaxis()->SetTitle("True Energy k_{B}T_{e} [keV]");
  ega2->GetYaxis()->SetTitle("#frac{Fit-Ref}{Ref}");
  float xx2[2] = {0, 100};
  float yy2[2] = {0, 0};
  auto line3 = new TGraph(2, xx2 , yy2);
  line3->Draw("Lsame");
  line3->SetLineColor(kCyan);
  line3->SetLineWidth(3);

}



void Draw_Incident_Spectrum_Fit(float Norm, float E1, float E2, float solid_angle)
{

  string seN, seNN;
  float Eval =0;

  A = A/solid_angle;
  err_A = err_A/solid_angle;

  f1 = new TF1("f1", "[0]*exp(-x/[1])", 0, 1000);
  f1->SetParameter(0, A);
  f1->SetParameter(1, E0);

  TF1* f2 = new TF1("f2", "[0]*exp(-x/[1]) + [2]*exp(-x/[3])", 0, 1000);
  f2->SetParameter(0, Acold);
  f2->SetParameter(1, Ecold);
  f2->SetParameter(2, Ahot);
  f2->SetParameter(3, Ehot);
  if (Chi21T <= Chi22T) Eval = f1->Integral(0, 1000);
  if (Chi22T < Chi21T) Eval = f2->Integral(0, 1000);

  TF1* f0 = new TF1("f0", "[0]*([1]*exp(-x/[2]) + exp(-x/[3]))", 0, 1000);
  f0->SetParameter(0, 1);
  f0->SetParameter(1, Norm);
  f0->SetParameter(2, E1);
  f0->SetParameter(3, E2);
  f0->SetNpx(200);
  f0->Draw("");
  f0->SetLineColor(kBlack);
  f0->SetMinimum(1);
  float integral = f0->Integral(0, 1000);
  f0->SetParameter(0, Eval/integral);
  
  f1->SetNpx(200);
  f1->Draw("same");
  f1->SetLineColor(kRed);
  f1->SetLineWidth(3);

  f2->SetNpx(200);
  f2->Draw("same");
  f2->SetLineColor(kBlue);
  f2->SetLineWidth(2);




  
  float Max = f1->GetMaximum();
  N = f1->Integral(0, 1000);
  eN = (N*(err_E0/E0 + err_A/A));
  
  seNN = Form("+-  %g #gamma/sr", eN);

  
  string sE = Form("Eo =  %g ", E0);
  TLatex *lE = new TLatex(400, 1*Max, sE.c_str());
  lE->Draw();
  lE->SetTextSize(0.035);
  lE->SetTextColor(kRed);

  string seE = Form("+-  %g keV", err_E0);
  TLatex *leE = new TLatex(700, 1*Max, seE.c_str());
  leE->Draw();
  leE->SetTextSize(0.035);
  leE->SetTextColor(kRed);

  string sNN = Form("N =  %g ", N);
  TLatex *lNN = new TLatex(400, 0.5*Max, sNN.c_str());
  lNN->Draw();
  lNN->SetTextSize(0.035);
  lNN->SetTextColor(kRed);

  TLatex *leNN = new TLatex(700, 0.5*Max, seNN.c_str());
  leNN->Draw();
  leNN->SetTextSize(0.035);
  leNN->SetTextColor(kRed);

  string sChi = Form("Chi2 =  %g ", Chi21T);
  TLatex *lChi = new TLatex(700, 0.2*Max, sChi.c_str());
  lChi->Draw();
  lChi->SetTextSize(0.035);
  lChi->SetTextColor(kRed);

  f1->GetYaxis()->SetTitle("dN/dE [keV^{-1} sr^{-1}]");
  f1->GetXaxis()->SetTitle("Energy [keV]");


  string sEcold= Form("E_{cold} =  %g ", Ecold);
  TLatex *lEcold = new TLatex(400,0.01*Max, sEcold.c_str());
  lEcold->Draw();
  lEcold->SetTextSize(0.035);
  lEcold->SetTextColor(kBlue);

  string seEcold = Form("+-  %g keV", err_Ecold);
  TLatex *leEcold = new TLatex(700, 0.01*Max, seEcold.c_str());
  leEcold->Draw();
  leEcold->SetTextSize(0.035);
  leEcold->SetTextColor(kBlue);

  string sEhot= Form("E_{hot} =  %g ", Ehot);
  TLatex *lEhot = new TLatex(400,0.005*Max, sEhot.c_str());
  lEhot->Draw();
  lEhot->SetTextSize(0.035);
  lEhot->SetTextColor(kBlue);

  string seEhot = Form("+-  %g keV", err_Ehot);
  TLatex *leEhot = new TLatex(700, 0.005*Max, seEhot.c_str());
  leEhot->Draw();
  leEhot->SetTextSize(0.035);
  leEhot->SetTextColor(kBlue);
  
  float N2 = f2->Integral(0, 1000);
  float eN2 = (N2*(err_Ecold/Ecold + err_Ehot/Ehot + err_Acold/Acold + err_Ahot/Ahot));


  string sNN2 = Form("N =  %g ", N2);
  TLatex *lNN2 = new TLatex(400, 0.002*Max, sNN2.c_str());
  lNN2->Draw();
  lNN2->SetTextSize(0.035);
  lNN2->SetTextColor(kBlue);

  string seNN2 = Form("+-  %g #gamma/sr", eN2);
  TLatex *leNN2 = new TLatex(700, 0.002*Max, seNN2.c_str());
  leNN2->Draw();
  leNN2->SetTextSize(0.035);
  leNN2->SetTextColor(kBlue);

  string sChi2 = Form("Chi2 =  %g ", Chi22T);
  TLatex *lChi2 = new TLatex(700, 0.0008*Max, sChi2.c_str());
  lChi2->Draw();
  lChi2->SetTextSize(0.035);
  lChi2->SetTextColor(kBlue);

}

