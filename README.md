# NeuroDB
This repository contains the implementation of NeuroDB to answer RAQs, k-th nearest neighbour and distance to k-th nearest neighbour queries. It contains a python module tha trains NeuroDB and a c++ module that loads a trained model to answer new queries. 

## Instalation and requirements
Run make to compile the c++ module for testing. It's been tested with g++ 7.5 and requires c++11. To train the model, the python module requires tensorflow, numpy, pandas and sklearn. It has been tested with tesorflow 2.4.0, numpy 1.19.5, pandas 1.0.1 and sklearn 0.22.1.

## Running NeuroDB
Running NeuroDB requires first craeting a config file containing all the configurations required and then calling python main.py to train the model. The program automatically generates training and testing data for a given query type and dataset, trains NeuroDB on the training data and runs the c++ module for testing.

## Config file
The file default\_config.py contains the default configuration and explanation for each parameter. Running python default\_config.py creates a json file containing default values.

## Examples
run\_RAQ.py shows an example of how config file should be generated for an RAQ. It trains and tests NeuroDB on the pm25.npy dataset in folder sample\_datasets. The generated training and test sets, together with the output of the experiment will be save in directory tests/pm25\_RAQ/. Calling python run\_RAQ.py should result in the following output saved in tests/pm25\_RAQ/out.txt

Starting data collection
Data collection took 3.98s
training model no 0 --------------
Epoch no 100 loss 0.17574083805084229 val_loss 0.31541430950164795 mae 0.3359383085479809 val_mae 0.42472206673971036
training model no 1 --------------
Epoch no 100 loss 0.16215768456459045 val_loss 0.16974817216396332 mae 0.3097802749235087 val_mae 0.31519522318025917
training model no 2 --------------
Epoch no 100 loss 0.17353861033916473 val_loss 0.22784963250160217 mae 0.3326042945164918 val_mae 0.3590013457507622
training model no 3 --------------
Epoch no 100 loss 0.018052691593766212 val_loss 0.01696583814918995 mae 0.10604410340260342 val_mae 0.10682268840510671
training model no 4 --------------
Epoch no 100 loss 0.180335134267807 val_loss 0.12096249312162399 mae 0.2823021602767441 val_mae 0.23965528534679878
training model no 5 --------------
Epoch no 100 loss 6.944283962249756 val_loss 5.200614929199219 mae 1.492409718015402 val_mae 1.2892991973132621
training model no 6 --------------
Epoch no 100 loss 0.1508488655090332 val_loss 0.1867433339357376 mae 0.3182021167961277 val_mae 0.3511426972179878
training model no 7 --------------
Epoch no 100 loss 3.809950113296509 val_loss 5.109820365905762 mae 1.4466931003212153 val_mae 1.8868177460461129
training model no 8 --------------
Epoch no 100 loss 0.3009437918663025 val_loss 0.25784337520599365 mae 0.39070208206067286 val_mae 0.407216514029154
training model no 9 --------------
Epoch no 100 loss 0.24778436124324799 val_loss 0.4998728930950165 mae 0.36588365070447 val_mae 0.4007374833269817
training model no 10 --------------
Epoch no 100 loss 0.2672102451324463 val_loss 0.428535133600235 mae 0.41679096997268805 val_mae 0.523200058355564
training model no 11 --------------
Epoch no 100 loss 0.1885354071855545 val_loss 0.2884461283683777 mae 0.3019447228145737 val_mae 0.36673569097751524
training model no 12 --------------
Epoch no 100 loss 2.3659169673919678 val_loss 2.7392797470092773 mae 1.1342792410647564 val_mae 1.1701369867092226
training model no 13 --------------
Epoch no 100 loss 0.15444332361221313 val_loss 0.19494925439357758 mae 0.31518337371986715 val_mae 0.35616953780011434
training model no 14 --------------
Epoch no 100 loss 0.2946549355983734 val_loss 0.33273670077323914 mae 0.45390866104921396 val_mae 0.48439993509432167
training model no 15 --------------
Epoch no 100 loss 0.1339171826839447 val_loss 0.1396588385105133 mae 0.28272440913531954 val_mae 0.2798960615948933
Model Training took 398.41s
Testing model
time:17.74989891 micro seconds
rmse:0.4707694948
avg rel acc:0.0004614487989
normalized rmse:0.0004622933338
mean result norm:1018.335022
model size:377.9375KB


run\_distNN.py shows an example of how config file should be generated for a distance to nearest neighbour query. It trains and tests NeuroDB on the gv25\_100k.npy dataset in folder sample\_datasets. The generated training and test sets, together with the output of the experiment will be save in directory tests/gv25\_10k\_distNN/. Calling python run\_distNN.py should result in the following output saved in tests/gv25\_10k\_distNN/out.txt

Starting data collection
Data collection took 4.18s
training model no 0 --------------
Epoch no 100 loss 0.002288664225488901 val_loss 0.001818509423173964 mae 0.037023301769352555 val_mae 0.032014946142832436
training model no 1 --------------
Epoch no 100 loss 0.01327413972467184 val_loss 0.005559703800827265 mae 0.09309180315185646 val_mae 0.06738315025965373
training model no 2 --------------
Epoch no 100 loss 0.0017961738631129265 val_loss 0.0032153399661183357 mae 0.03176682700151167 val_mae 0.04679874579111735
training model no 3 --------------
Epoch no 100 loss 0.008066701702773571 val_loss 0.0017443207325413823 mae 0.07328015828282577 val_mae 0.03473069270451864
training model no 4 --------------
Epoch no 100 loss 0.0008311241399496794 val_loss 0.0006309655145741999 mae 0.021368254280690127 val_mae 0.020634313424428303
training model no 5 --------------
Epoch no 100 loss 0.09551838785409927 val_loss 0.02856202982366085 mae 0.24798715639414284 val_mae 0.14199296633402506
training model no 6 --------------
Epoch no 100 loss 0.005137626081705093 val_loss 0.006146854255348444 mae 0.054363419424812744 val_mae 0.06554919481277466
training model no 7 --------------
Epoch no 100 loss 0.007129133678972721 val_loss 0.008502942509949207 mae 0.06235475210273788 val_mae 0.06387211879094441
training model no 8 --------------
Epoch no 100 loss 0.004829985089600086 val_loss 0.007847688160836697 mae 0.05659184515851099 val_mae 0.08159518241882324
training model no 9 --------------
Epoch no 100 loss 0.0029824464581906796 val_loss 0.0016623918199911714 mae 0.04174072075190033 val_mae 0.03413361310958862
training model no 10 --------------
Epoch no 100 loss 0.004609445575624704 val_loss 0.004577321466058493 mae 0.05701213212882948 val_mae 0.05637268225351969
training model no 11 --------------
Epoch no 100 loss 0.028000978752970695 val_loss 0.007074467837810516 mae 0.1284703308681272 val_mae 0.06803729136784871
training model no 12 --------------
Epoch no 100 loss 0.038076236844062805 val_loss 0.06937713176012039 mae 0.1623488293503815 val_mae 0.20955008268356323
training model no 13 --------------
Epoch no 100 loss 0.022435078397393227 val_loss 0.0326092503964901 mae 0.12002047352820823 val_mae 0.1439306139945984
training model no 14 --------------
Epoch no 100 loss 0.0009294074261561036 val_loss 0.0005845865816809237 mae 0.02183964192492407 val_mae 0.0167236328125
training model no 15 --------------
Epoch no 100 loss 0.002784490119665861 val_loss 0.01159862894564867 mae 0.04087308967638315 val_mae 0.09136253595352173
Model Training took 403.35s
Testing model
time:22.28580093 micro seconds
rmse:0.07242497802
avg rel acc:0.01817158423
normalized rmse:0.01787892357
mean result norm:4.050858021
model size:449.1875KB
