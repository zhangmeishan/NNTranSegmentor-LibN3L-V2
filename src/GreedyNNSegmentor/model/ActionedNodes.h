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
	LookupNode last_word_input;
	LookupNode last2_word_input;
	BiNode word_conv;
	IncLSTM1Builder word_lstm;

	LookupNode last_action_input;
	LookupNode last2_action_input;
	BiNode action_conv;
	IncLSTM1Builder action_lstm;
	
	FourNode sep_hidden;
	TriNode app_hidden;
	LinearNode sep_score;
	LinearNode app_score;

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

	vector<SPAddNode> outputs;

	Node bucket;

public:
	inline void initial(ModelParams& params, HyperParams& hyparams){
		//neural features
		last_word_input.setParam(&(params.word_table));
		last_word_input.setDropout(hyparams.dropProb);
		last2_word_input.setParam(&(params.word_table));
		last2_word_input.setDropout(hyparams.dropProb);
		word_conv.setParam(&(params.word_conv));
		word_conv.setDropout(hyparams.dropProb);
		word_lstm.setParam(&(params.word_lstm), hyparams.dropProb);

		last_action_input.setParam(&(params.action_table));
		last_action_input.setDropout(hyparams.dropProb);
		last2_action_input.setParam(&(params.action_table));
		last2_action_input.setDropout(hyparams.dropProb);
		action_conv.setParam(&(params.action_conv));
		action_conv.setDropout(hyparams.dropProb);
		action_lstm.setParam(&(params.action_lstm), hyparams.dropProb);

		sep_hidden.setParam(&(params.sep_hidden));
		sep_hidden.setDropout(hyparams.dropProb);
		app_hidden.setParam(&(params.app_hidden));
		app_hidden.setDropout(hyparams.dropProb);
		sep_score.setParam(&(params.sep_score));
		app_score.setParam(&(params.app_score));

		//sparse features
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

		outputs.resize(hyparams.action_num);

		bucket.val = Mat::Zero(hyparams.char_lstm_dim, 1);
	}


public:
	inline void forward(Graph* cg, const vector<CAction>& actions, const AtomFeatures& atomFeat){
		static vector<PNode> sumNodes;
		static CAction ac;
		static int ac_num;
		ac_num = actions.size();

		last2_action_input.forward(cg, atomFeat.str_2AC);
		last_action_input.forward(cg, atomFeat.str_1AC);
		action_conv.forward(cg, &last2_action_input, &last_action_input);
		action_lstm.forward(cg, &action_conv, atomFeat.p_action_lstm);

		last2_word_input.forward(cg, atomFeat.str_2W);
		last_word_input.forward(cg, atomFeat.str_1W);
		word_conv.forward(cg, &last2_word_input, &last_word_input);
		word_lstm.forward(cg, &word_conv, atomFeat.p_word_lstm);

		PNode P_char_left_lstm = atomFeat.next_position >= 0 ? &(atomFeat.p_char_left_lstm->_hiddens[atomFeat.next_position]) : &bucket;
		PNode P_char_right_lstm = atomFeat.next_position >= 0 ? &(atomFeat.p_char_right_lstm->_hiddens[atomFeat.next_position]) : &bucket;


		for (int idx = 0; idx < ac_num; idx++){
			ac.set(actions[idx]._code);
			sumNodes.clear();
			/*
			if (ac.isAppend()){
				app_1C_C0.forward(cg, atomFeat.sid_1C, atomFeat.sid_C0);
				if (app_1C_C0.executed)sumNodes.push_back(&app_1C_C0);

				app_1Wc0_C0.forward(cg, atomFeat.sid_1Wc0, atomFeat.sid_C0);
				if (app_1Wc0_C0.executed)sumNodes.push_back(&app_1Wc0_C0);

				app_2CT_1CT_CT0.forward(cg, atomFeat.sid_2CT, atomFeat.sid_1CT, atomFeat.sid_CT0);
				if (app_2CT_1CT_CT0.executed)sumNodes.push_back(&app_2CT_1CT_CT0);
			}
			else{
				sep_1C_C0.forward(cg, atomFeat.sid_1C, atomFeat.sid_C0);
				if (sep_1C_C0.executed)sumNodes.push_back(&sep_1C_C0);

				sep_1Wc0_C0.forward(cg, atomFeat.sid_1Wc0, atomFeat.sid_C0);
				if (sep_1Wc0_C0.executed)sumNodes.push_back(&sep_1Wc0_C0);

				sep_2CT_1CT_CT0.forward(cg, atomFeat.sid_2CT, atomFeat.sid_1CT, atomFeat.sid_CT0);
				if (sep_2CT_1CT_CT0.executed)sumNodes.push_back(&sep_2CT_1CT_CT0);

				sep_1W.forward(cg, atomFeat.sid_1W);
				if (sep_1W.executed)sumNodes.push_back(&sep_1W);

				sep_1WD_1WL.forward(cg, atomFeat.sid_1WD, atomFeat.sid_1WL);
				if (sep_1WD_1WL.executed)sumNodes.push_back(&sep_1WD_1WL);

				if (atomFeat.sid_1WL == 1){
					sep_1WSingle.forward(cg, atomFeat.sid_1W);
					if (sep_1WSingle.executed)sumNodes.push_back(&sep_1WSingle);
				}

				sep_1W_C0.forward(cg, atomFeat.sid_1W, atomFeat.sid_C0);
				if (sep_1W_C0.executed)sumNodes.push_back(&sep_1W_C0);

				sep_2W_1W.forward(cg, atomFeat.sid_2W, atomFeat.sid_1W);
				if (sep_2W_1W.executed)sumNodes.push_back(&sep_2W_1W);

				sep_2Wc0_1W.forward(cg, atomFeat.sid_2Wc0, atomFeat.sid_1W);
				if (sep_2Wc0_1W.executed)sumNodes.push_back(&sep_2Wc0_1W);

				sep_2Wcn_1W.forward(cg, atomFeat.sid_2Wcn, atomFeat.sid_1W);
				if (sep_2Wcn_1W.executed)sumNodes.push_back(&sep_2Wcn_1W);

				sep_2Wc0_1Wc0.forward(cg, atomFeat.sid_2Wc0, atomFeat.sid_1Wc0);
				if (sep_2Wc0_1Wc0.executed)sumNodes.push_back(&sep_2Wc0_1Wc0);

				sep_2Wcn_1Wcn.forward(cg, atomFeat.sid_2Wcn, atomFeat.sid_1C);
				if (sep_2Wcn_1Wcn.executed)sumNodes.push_back(&sep_2Wcn_1Wcn);

				sep_2W_1WL.forward(cg, atomFeat.sid_2W, atomFeat.sid_1WL);
				if (sep_2W_1WL.executed)sumNodes.push_back(&sep_2W_1WL);

				sep_2WL_1W.forward(cg, atomFeat.sid_2WL, atomFeat.sid_1W);
				if (sep_2WL_1W.executed)sumNodes.push_back(&sep_2WL_1W);

				sep_2W_1Wcn.forward(cg, atomFeat.sid_2W, atomFeat.sid_1C);
				if (sep_2W_1Wcn.executed)sumNodes.push_back(&sep_2W_1Wcn);

				sep_1Wc0_1WL.forward(cg, atomFeat.sid_1Wc0, atomFeat.sid_1WL);
				if (sep_1Wc0_1WL.executed)sumNodes.push_back(&sep_1Wc0_1WL);

				sep_1Wcn_1WL.forward(cg, atomFeat.sid_1C, atomFeat.sid_1WL);
				if (sep_1Wcn_1WL.executed)sumNodes.push_back(&sep_1Wcn_1WL);

				for (int idx = 0; idx < atomFeat.sid_1Wci.size(); idx++){
					sep_1Wci_1Wcn[idx].forward(cg, atomFeat.sid_1Wci[idx], atomFeat.sid_1C);
					if (sep_1Wci_1Wcn[idx].executed)sumNodes.push_back(&(sep_1Wci_1Wcn[idx]));
				}
			}
			*/
			/*if (prevStateNode != NULL){
				sumNodes.push_back(prevStateNode);
			}*/

			//neural features

			if (ac.isAppend()){
				app_hidden.forward(cg, P_char_left_lstm, P_char_right_lstm, &(action_lstm._hidden));
				app_score.forward(cg, &app_hidden);
				sumNodes.push_back(&app_score);
			}
			else{
				sep_hidden.forward(cg, P_char_left_lstm, P_char_right_lstm, &(word_lstm._hidden), &(action_lstm._hidden));
				sep_score.forward(cg, &sep_hidden);
				sumNodes.push_back(&sep_score);
			}

			outputs[ac._code].forward(cg, sumNodes, ac._code);
		}
	}

};

#endif /* SRC_ActionedNodes_H_ */
