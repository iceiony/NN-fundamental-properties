 {
     printf("Load dynet libraries\n");

     gROOT->ProcessLine(".include ./dynet");
     gROOT->ProcessLine(".L ./dynet/libdynet.dylib");
 }
