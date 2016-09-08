/*
 * Feature.h
 *
 *  Created on: Aug 25, 2016
 *      Author: mszhang
 */

#ifndef SRC_ActionedNodes_H_
#define SRC_ActionedNodes_H_

#include "ModelParams.h"
#include "AtomFeatures.h"

struct ActionedNodes {

	//append feature parameters
	SparseC2Node  app_1C_C0;
	SparseC2Node  app_1Wc0_C0;
	SparseC3Node  app_2CT_1CT_CT0;

	//separate feature parameters
	SparseC2Node  sep_1C_C0;
	SparseC2Node  sep_1Wc0_C0;
	SparseC3Node  sep_2CT_1CT_CT0;
	SparseC1Node  sep_1W;
	SparseC2Node  sep_1WD_1WL;
	SparseC1Node  sep_1WSingle;
	SparseC2Node  sep_1W_C0;
	SparseC2Node  sep_2W_1W;
	SparseC2Node  sep_2Wc0_1W;
	SparseC2Node  sep_2Wcn_1W;
	SparseC2Node  sep_2Wc0_1Wc0;
	SparseC2Node  sep_2Wcn_1Wcn;
	SparseC2Node  sep_2W_1WL;
	SparseC2Node  sep_2WL_1W;
	SparseC2Node  sep_2W_1Wcn;
	SparseC2Node  sep_1Wc0_1WL;
	SparseC2Node  sep_1Wcn_1WL;
	vector<SparseC2Node>  sep_1Wci_1Wcn;

	PAddNode output;

public:
	inline void initial(ModelParams& params, HyperParams& hyparams){
		app_1C_C0.setParam(&params.app_1C_C0);
		app_1Wc0_C0.setParam(&params.app_1Wc0_C0);
		app_2CT_1CT_CT0.setParam(&params.app_2CT_1CT_CT0);

		sep_1C_C0.setParam(&params.sep_1C_C0);
		sep_1Wc0_C0.setParam(&params.sep_1Wc0_C0);
		sep_2CT_1CT_CT0.setParam(&params.sep_2CT_1CT_CT0);
		sep_1W.setParam(&params.sep_1W);
		sep_1WD_1WL.setParam(&params.sep_1WD_1WL);
		sep_1WSingle.setParam(&params.sep_1WSingle);
		sep_1W_C0.setParam(&params.sep_1W_C0);
		sep_2W_1W.setParam(&params.sep_2W_1W);
		sep_2Wc0_1W.setParam(&params.sep_2Wc0_1W);
		sep_2Wcn_1W.setParam(&params.sep_2Wcn_1W);
		sep_2Wc0_1Wc0.setParam(&params.sep_2Wc0_1Wc0);
		sep_2Wcn_1Wcn.setParam(&params.sep_2Wcn_1Wcn);
		sep_2W_1WL.setParam(&params.sep_2W_1WL);
		sep_2WL_1W.setParam(&params.sep_2WL_1W);
		sep_2W_1Wcn.setParam(&params.sep_2W_1Wcn);
		sep_1Wc0_1WL.setParam(&params.sep_1Wc0_1WL);
		sep_1Wcn_1WL.setParam(&params.sep_1Wcn_1WL);
		sep_1Wci_1Wcn.resize(hyparams.maxlength);
		for (int idx = 0; idx < sep_1Wci_1Wcn.size(); idx++){
			sep_1Wci_1Wcn[idx].setParam(&params.sep_1Wci_1Wcn);
		}

	}


public:
	inline void forward(Graph* cg, const CAction& ac, const AtomFeatures& atomFeat, PNode prevStateNode){
		static vector<PNode> sumNodes;
		sumNodes.clear();
		if (ac.isAppend()){
			app_1C_C0.forward(cg, atomFeat.str_1C, atomFeat.str_C0);
			sumNodes.push_back(&app_1C_C0);

			app_1Wc0_C0.forward(cg, atomFeat.str_1Wc0, atomFeat.str_C0);
			sumNodes.push_back(&app_1Wc0_C0);

			app_2CT_1CT_CT0.forward(cg, atomFeat.str_2CT, atomFeat.str_1CT, atomFeat.str_CT0);
			sumNodes.push_back(&app_2CT_1CT_CT0);
		}
		else{
			sep_1C_C0.forward(cg, atomFeat.str_1C, atomFeat.str_C0);
			sumNodes.push_back(&sep_1C_C0);

			sep_1Wc0_C0.forward(cg, atomFeat.str_1Wc0, atomFeat.str_C0);
			sumNodes.push_back(&sep_1Wc0_C0);

			sep_2CT_1CT_CT0.forward(cg, atomFeat.str_2CT, atomFeat.str_1CT, atomFeat.str_CT0);
			sumNodes.push_back(&sep_2CT_1CT_CT0);

			sep_1W.forward(cg, atomFeat.str_1W);
			sumNodes.push_back(&sep_1W);

			sep_1WD_1WL.forward(cg, atomFeat.str_1WD, atomFeat.str_1WL);
			sumNodes.push_back(&sep_1WD_1WL);

			if (atomFeat.int_1WL == 1){
				sep_1WSingle.forward(cg, atomFeat.str_1W);
				sumNodes.push_back(&sep_1WSingle);
			}

			sep_1W_C0.forward(cg, atomFeat.str_1W, atomFeat.str_C0);
			sumNodes.push_back(&sep_1W_C0);

			sep_2W_1W.forward(cg, atomFeat.str_2W, atomFeat.str_1W);
			sumNodes.push_back(&sep_2W_1W);

			sep_2Wc0_1W.forward(cg, atomFeat.str_2Wc0, atomFeat.str_1W);
			sumNodes.push_back(&sep_2Wc0_1W);

			sep_2Wcn_1W.forward(cg, atomFeat.str_2Wcn, atomFeat.str_1W);
			sumNodes.push_back(&sep_2Wcn_1W);

			sep_2Wc0_1Wc0.forward(cg, atomFeat.str_2Wc0, atomFeat.str_1Wc0);
			sumNodes.push_back(&sep_2Wc0_1Wc0);

			sep_2Wcn_1Wcn.forward(cg, atomFeat.str_2Wcn, atomFeat.str_1C);
			sumNodes.push_back(&sep_2Wcn_1Wcn);

			sep_2W_1WL.forward(cg, atomFeat.str_2W, atomFeat.str_1WL);
			sumNodes.push_back(&sep_2W_1WL);

			sep_2WL_1W.forward(cg, atomFeat.str_2WL, atomFeat.str_1W);
			sumNodes.push_back(&sep_2WL_1W);

			sep_2W_1Wcn.forward(cg, atomFeat.str_2W, atomFeat.str_1C);
			sumNodes.push_back(&sep_2W_1Wcn);

			sep_1Wc0_1WL.forward(cg, atomFeat.str_1Wc0, atomFeat.str_1WL);
			sumNodes.push_back(&sep_1Wc0_1WL);

			sep_1Wcn_1WL.forward(cg, atomFeat.str_1C, atomFeat.str_1WL);
			sumNodes.push_back(&sep_1Wcn_1WL);

			for (int idx = 0; idx < atomFeat.str_1Wci.size(); idx++){
				sep_1Wci_1Wcn[idx].forward(cg, atomFeat.str_1Wci[idx], atomFeat.str_1C);
				sumNodes.push_back(&(sep_1Wci_1Wcn[idx]));
			}
		}

		if (prevStateNode != NULL){
			sumNodes.push_back(prevStateNode);
		}

		output.forward(cg, sumNodes);
	}

};

#endif /* SRC_ActionedNodes_H_ */
