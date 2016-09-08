#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_
#include "HyperParams.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams{

public:
	//should be initialized outside
	Alphabet words; // words
	Alphabet chars; // chars
	Alphabet charTypes; // char type
	Alphabet wordLengths; // word length
	Alphabet dictionarys; // lexicon features: iv or oov words
	Alphabet actions; // words


	//feature templates
	//append feature parameters
	SparseC2Params  app_1C_C0;
	SparseC2Params  app_1Wc0_C0;
	SparseC3Params  app_2CT_1CT_CT0;

	//separate feature parameters
	SparseC2Params  sep_1C_C0;
	SparseC2Params  sep_1Wc0_C0;
	SparseC3Params  sep_2CT_1CT_CT0;
	SparseC1Params  sep_1W;
	SparseC2Params  sep_1WD_1WL;
	SparseC1Params  sep_1WSingle;
	SparseC2Params  sep_1W_C0;
	SparseC2Params  sep_2W_1W;
	SparseC2Params  sep_2Wc0_1W;
	SparseC2Params  sep_2Wcn_1W;
	SparseC2Params  sep_2Wc0_1Wc0;
	SparseC2Params  sep_2Wcn_1Wcn;
	SparseC2Params  sep_2W_1WL;
	SparseC2Params  sep_2WL_1W;
	SparseC2Params  sep_2W_1Wcn;
	SparseC2Params  sep_1Wc0_1WL;
	SparseC2Params  sep_1Wcn_1WL;
	SparseC2Params  sep_1Wci_1Wcn;


public:
	bool initial(HyperParams& opts){
		// some model parameters should be initialized outside
		if (words.size() <= 0 || chars.size() <= 0 || charTypes.size() <= 0 || wordLengths.size() <= 0 || dictionarys.size() <= 0){
			return false;
		}

		app_1C_C0.initial(&chars, &chars, 1);
		app_1Wc0_C0.initial(&chars, &chars, 1);
		app_2CT_1CT_CT0.initial(&charTypes, &charTypes, &charTypes, 1);

		sep_1C_C0.initial(&chars, &chars, 1);
		sep_1Wc0_C0.initial(&chars, &chars, 1);
		sep_2CT_1CT_CT0.initial(&charTypes, &charTypes, &charTypes, 1);
		sep_1W.initial(&words, 1);
		sep_1WD_1WL.initial(&dictionarys, &wordLengths, 1);
		sep_1WSingle.initial(&chars, 1);
		sep_1W_C0.initial(&words, &chars, 1);
		sep_2W_1W.initial(&words, &words, 1);
		sep_2Wc0_1W.initial(&chars, &words, 1);
		sep_2Wcn_1W.initial(&chars, &words, 1);
		sep_2Wc0_1Wc0.initial(&chars, &chars, 1);
		sep_2Wcn_1Wcn.initial(&chars, &chars, 1);
		sep_2W_1WL.initial(&words, &wordLengths, 1);
		sep_2WL_1W.initial(&wordLengths, &words, 1);
		sep_2W_1Wcn.initial(&words, &chars, 1);
		sep_1Wc0_1WL.initial(&chars, &wordLengths, 1);
		sep_1Wcn_1WL.initial(&chars, &wordLengths, 1);
		sep_1Wci_1Wcn.initial(&chars, &chars, 1);

		return true;
	}


	void exportModelParams(ModelUpdate& ada){
		app_1C_C0.exportAdaParams(ada);
		app_1Wc0_C0.exportAdaParams(ada);
		app_2CT_1CT_CT0.exportAdaParams(ada);

		sep_1C_C0.exportAdaParams(ada);
		sep_1Wc0_C0.exportAdaParams(ada);
		sep_2CT_1CT_CT0.exportAdaParams(ada);
		sep_1W.exportAdaParams(ada);
		sep_1WD_1WL.exportAdaParams(ada);
		sep_1WSingle.exportAdaParams(ada);
		sep_1W_C0.exportAdaParams(ada);
		sep_2W_1W.exportAdaParams(ada);
		sep_2Wc0_1W.exportAdaParams(ada);
		sep_2Wcn_1W.exportAdaParams(ada);
		sep_2Wc0_1Wc0.exportAdaParams(ada);
		sep_2Wcn_1Wcn.exportAdaParams(ada);
		sep_2W_1WL.exportAdaParams(ada);
		sep_2WL_1W.exportAdaParams(ada);
		sep_2W_1Wcn.exportAdaParams(ada);
		sep_1Wc0_1WL.exportAdaParams(ada);
		sep_1Wcn_1WL.exportAdaParams(ada);
		sep_1Wci_1Wcn.exportAdaParams(ada);
	}



	// will add it later
	void saveModel(){

	}

	void loadModel(const string& inFile){

	}

};

#endif /* SRC_ModelParams_H_ */