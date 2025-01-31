#include"Analyse.hh"

void Analyse()
{
  float Energy[n] = {1, 2, 4, 6, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 400, 500, 1000};

  for (int i=0; i<n; i++)
    {
      sprintf(file, "Config19/Config_19_%0.0fkeV_10M.root", Energy[i]);
      cout << "file = " << file << endl;
      h = GetGraph(file);
      IP1[i] = GetBinValue(1);
      IP2[i] = GetBinValue(2);
      IP3[i] = GetBinValue(3);
      IP4[i] = GetBinValue(4);
      IP5[i] = GetBinValue(5);
      IP6[i] = GetBinValue(6);
      IP7[i] = GetBinValue(7);
      IP8[i] = GetBinValue(8);
      IP9[i] = GetBinValue(9);
      IP10[i] = GetBinValue(10);
      IP11[i] = GetBinValue(11);
      IP12[i] = GetBinValue(12);
      IP13[i] = GetBinValue(13);
      IP14[i] = GetBinValue(14);
      IP15[i] = GetBinValue(15);

      
    }

  TGraph* gIP1 = new TGraph(n, Energy, IP1);
  gIP1 ->Draw("APC");
  gIP1 ->SetMarkerStyle(8);
  gIP1 ->SetLineWidth(3);

  TGraph* gIP2 = new TGraph(n, Energy, IP2);
  gIP2 ->Draw("samePC");
  gIP2 ->SetMarkerStyle(8);
  gIP2 ->SetMarkerColor(kRed);
  gIP2 ->SetLineColor(kRed);
  gIP2 ->SetLineWidth(3);

  TGraph* gIP3 = new TGraph(n, Energy, IP3);
  gIP3 ->Draw("samePC");
  gIP3 ->SetMarkerStyle(8);
  gIP3 ->SetMarkerColor(kBlue);
  gIP3 ->SetLineColor(kBlue);
  gIP3 ->SetLineWidth(3);

  TGraph* gIP4 = new TGraph(n, Energy, IP4);
  gIP4 ->Draw("samePC");
  gIP4 ->SetMarkerStyle(8);
  gIP4 ->SetMarkerColor(kCyan);
  gIP4 ->SetLineColor(kCyan);
  gIP4 ->SetLineWidth(3);

  TGraph* gIP5 = new TGraph(n, Energy, IP5);
  gIP5 ->Draw("samePC");
  gIP5 ->SetMarkerStyle(8);
  gIP5 ->SetMarkerColor(kGreen);
  gIP5 ->SetLineColor(kGreen);
  gIP5 ->SetLineWidth(3);

  TGraph* gIP6 = new TGraph(n, Energy, IP6);
  gIP6 ->Draw("samePC");
  gIP6 ->SetMarkerStyle(8);
  gIP6 ->SetMarkerColor(kOrange);
  gIP6 ->SetLineColor(kOrange);
  gIP6 ->SetLineWidth(3);

  TGraph* gIP7 = new TGraph(n, Energy, IP7);
  gIP7 ->Draw("samePC");
  gIP7 ->SetMarkerStyle(8);
  gIP7 ->SetMarkerColor(kGreen+2);
  gIP7 ->SetLineColor(kGreen+2);
  gIP7 ->SetLineWidth(3);

  TGraph* gIP8 = new TGraph(n, Energy, IP8);
  gIP8 ->Draw("samePC");
  gIP8 ->SetMarkerStyle(8);
  gIP8 ->SetMarkerColor(kGray);
  gIP8 ->SetLineColor(kGray);
  gIP8 ->SetLineWidth(3);
  gIP8 ->SetLineStyle(2);

  TGraph* gIP9 = new TGraph(n, Energy, IP9);
  gIP9 ->Draw("samePC");
  gIP9 ->SetMarkerStyle(8);
  gIP9 ->SetLineWidth(3);
  gIP9 ->SetLineStyle(2);

  TGraph* gIP10 = new TGraph(n, Energy, IP10);
  gIP10 ->Draw("samePC");
  gIP10 ->SetMarkerStyle(8);
  gIP10 ->SetMarkerColor(kRed);
  gIP10 ->SetLineColor(kRed);
  gIP10 ->SetLineWidth(3);
  gIP10 ->SetLineStyle(2);
  
  TGraph* gIP11 = new TGraph(n, Energy, IP11);
  gIP11 ->Draw("samePC");
  gIP11 ->SetMarkerStyle(8);
  gIP11 ->SetMarkerColor(kBlue);
  gIP11 ->SetLineColor(kBlue);
  gIP11 ->SetLineWidth(3);
  gIP11 ->SetLineStyle(2);

  TGraph* gIP12 = new TGraph(n, Energy, IP12);
  gIP12 ->Draw("samePC");
  gIP12 ->SetMarkerStyle(8);
  gIP12 ->SetMarkerColor(kCyan);
  gIP12 ->SetLineColor(kCyan);
  gIP12 ->SetLineWidth(3);
  gIP12 ->SetLineStyle(2);

  TGraph* gIP13 = new TGraph(n, Energy, IP13);
  gIP13 ->Draw("samePC");
  gIP13 ->SetMarkerStyle(8);
  gIP13 ->SetMarkerColor(kGreen);
  gIP13 ->SetLineColor(kGreen);
  gIP13 ->SetLineWidth(3);
  gIP13 ->SetLineStyle(2);

  TGraph* gIP14 = new TGraph(n, Energy, IP14);
  gIP14 ->Draw("samePC");
  gIP14 ->SetMarkerStyle(8);
  gIP14 ->SetMarkerColor(kOrange);
  gIP14 ->SetLineColor(kOrange);
  gIP14 ->SetLineWidth(3);
  gIP14 ->SetLineStyle(2);

  TGraph* gIP15 = new TGraph(n, Energy, IP15);
  gIP15 ->Draw("samePC");
  gIP15 ->SetMarkerStyle(8);
  gIP15 ->SetMarkerColor(kGreen+2);
  gIP15 ->SetLineColor(kGreen+2);
  gIP15 ->SetLineWidth(3);
  gIP15 ->SetLineStyle(2);


  gIP1->GetXaxis()->SetTitle("#gamma Energy [keV]");
  gIP1->GetXaxis()->SetTitleOffset(1.2);
  gIP1->GetXaxis()->CenterTitle(true);
  gIP1->GetYaxis()->SetTitle("keV/#gamma");
  gIP1->GetYaxis()->CenterTitle(true);
  gIP1->GetYaxis()->SetRangeUser(1e-7, 0.1);;


  TLegend* legend = new TLegend(0.1, 0.5, 0.3, 0.9);
  legend->AddEntry(gIP1, "IP1 [26 um Alu]", "PL");
  legend->AddEntry(gIP2, "IP2 [1 mm Alu]", "PL");
  legend->AddEntry(gIP3, "IP3 [4 mm Alu]", "PL");
  legend->AddEntry(gIP4, "IP4 [8 mm Alu]", "PL");
  legend->AddEntry(gIP5, "IP5 [1 mm Cu]", "PL");
  legend->AddEntry(gIP6, "IP6 [1 mm Cu]", "PL");
  legend->AddEntry(gIP7, "IP7 [3 mm Cu]", "PL");
  legend->AddEntry(gIP8, "IP8 [4 mm Cu]", "PL");
  legend->AddEntry(gIP9, "IP9 [2 mm Sn]", "PL");
  legend->AddEntry(gIP10, "IP10 [3 mm Sn]", "PL");
  legend->AddEntry(gIP11, "IP11 [2 mm Pb]", "PL");
  legend->AddEntry(gIP12, "IP12 [2 mm Pb]", "PL");
  legend->AddEntry(gIP13, "IP13 [4 mm Pb]", "PL");
  legend->AddEntry(gIP14, "IP14 [8 mm Pb]", "PL");
  legend->AddEntry(gIP15, "IP15 [12 mm Pb]", "PL");
  legend->Draw();


}
