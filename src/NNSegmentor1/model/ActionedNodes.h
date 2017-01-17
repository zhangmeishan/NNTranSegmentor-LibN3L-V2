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
#include "Action.h"

struct ActionedNodes {
	LookupNode last_word_input;
	LookupNode last2_word_input;
	BiNode word_conv;
	IncLSTM1Builder word_lstm;
	
	TriNode sep_hidden;
	BiNode app_hidden;
	LinearNode sep_score;
	LinearNode app_score;

	vector<SPAddNode> outputs;

	Node bucket;

public:
	inline void initial(ModelParams& params, HyperParams& hyparams, AlignedMemoryPool* mem){
		//neural features
		last_word_input.setParam(&(params.word_table));
		last2_word_input.setParam(&(params.word_table));
		word_conv.setParam(&(params.word_conv));
		word_lstm.init(&(params.word_lstm), hyparams.dropProb, mem); //already allocated here


		sep_hidden.setParam(&(params.sep_hidden));
		app_hidden.setParam(&(params.app_hidden));
		sep_score.setParam(&(params.sep_score));
		app_score.setParam(&(params.app_score));


		outputs.resize(hyparams.action_num);

		//allocate node memories
		last_word_input.init(hyparams.word_dim, hyparams.dropProb, mem);
		last2_word_input.init(hyparams.word_dim, hyparams.dropProb, mem);
		word_conv.init(hyparams.word_hidden_dim, hyparams.dropProb, mem);
		
		sep_hidden.init(hyparams.sep_hidden_dim, -1, mem);
		app_hidden.init(hyparams.app_hidden_dim, -1, mem);
		sep_score.init(1, -1, mem);
		app_score.init(1, -1, mem);
		
		bucket.init(hyparams.char_lstm_dim, -1, mem);
        bucket.set_bucket();
		
		for (int idx = 0; idx < hyparams.action_num; idx++) {
			outputs[idx].init(1, -1, mem);
		}
	}


public:
	inline void forward(Graph* cg, const vector<CAction>& actions, const AtomFeatures& atomFeat, PNode prevStateNode){
		static vector<PNode> sumNodes;
		static CAction ac;
		static int ac_num;
		ac_num = actions.size();

		last2_word_input.forward(cg, atomFeat.str_2W);
		last_word_input.forward(cg, atomFeat.str_1W);
		word_conv.forward(cg, &last2_word_input, &last_word_input);
		word_lstm.forward(cg, &word_conv, atomFeat.p_word_lstm);

		PNode P_char_left_lstm = atomFeat.next_position >= 0 ? &(atomFeat.p_char_left_lstm->_hiddens[atomFeat.next_position]) : &bucket;
		PNode P_char_right_lstm = atomFeat.next_position >= 0 ? &(atomFeat.p_char_right_lstm->_hiddens[atomFeat.next_position]) : &bucket;


		for (int idx = 0; idx < ac_num; idx++){
			ac.set(actions[idx]);
			sumNodes.clear();

			if (prevStateNode != NULL){
				sumNodes.push_back(prevStateNode);
			}

			//neural features
			if (ac.isAppend()){
				app_hidden.forward(cg, P_char_left_lstm, P_char_right_lstm);
				app_score.forward(cg, &app_hidden);
				sumNodes.push_back(&app_score);
			}
			else{
				sep_hidden.forward(cg, P_char_left_lstm, P_char_right_lstm, &(word_lstm._hidden));
				sep_score.forward(cg, &sep_hidden);
				sumNodes.push_back(&sep_score);
			}

			outputs[idx].forward(cg, sumNodes, 0);
		}
	}

};

#endif /* SRC_ActionedNodes_H_ */
