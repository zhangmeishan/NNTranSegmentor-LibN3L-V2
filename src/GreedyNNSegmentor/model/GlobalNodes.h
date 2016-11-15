/*
 * Feature.h
 *
 *  Created on: Aug 25, 2016
 *      Author: mszhang
 */

#ifndef SRC_GlobalNodes_H_
#define SRC_GlobalNodes_H_

#include "ModelParams.h"

struct GlobalNodes {
	vector<LookupNode> char_inputs;
	vector<LookupNode> bichar_inputs;
	vector<ConcatNode> char_repsents;
	WindowBuilder char_window;
	vector<UniNode> char_tanh_conv;
	LSTM1Builder char_left_lstm;
	LSTM1Builder char_right_lstm;

public:
	inline void resize(int max_length){
		char_inputs.resize(max_length);
		bichar_inputs.resize(max_length);
		char_repsents.resize(max_length);
		char_window.resize(max_length);
		char_tanh_conv.resize(max_length);
		char_left_lstm.resize(max_length);
		char_right_lstm.resize(max_length);
	}

public:
	inline void initial(ModelParams& params, HyperParams& hyparams){
		int length = char_inputs.size();
		for (int idx = 0; idx < length; idx++){
			char_inputs[idx].setParam(&params.char_table);
			bichar_inputs[idx].setParam(&params.bichar_table);
			char_repsents[idx].setDropout(hyparams.dropProb);			
			char_tanh_conv[idx].setParam(&params.char_tanh_conv);
			char_tanh_conv[idx].setDropout(hyparams.dropProb);
		}
		char_window.setContext(hyparams.char_context);
		char_left_lstm.setParam(&params.char_left_lstm, hyparams.dropProb, true);
		char_right_lstm.setParam(&params.char_right_lstm, hyparams.dropProb, true);
	}


public:
	inline void forward(Graph* cg, const std::vector<std::string>* pCharacters){
		int char_size = pCharacters->size();
		string unichar, biChar;
		for (int idx = 0; idx < char_size; idx++){
			unichar = (*pCharacters)[idx];
			biChar = idx < char_size - 1 ? (*pCharacters)[idx] + (*pCharacters)[idx + 1] : nullkey;
			char_inputs[idx].forward(cg, unichar);
			bichar_inputs[idx].forward(cg, biChar);
			char_repsents[idx].forward(cg, &char_inputs[idx], &bichar_inputs[idx]);
		}
		char_window.forward(cg, getPNodes(char_repsents, char_size));
		for (int idx = 0; idx < char_size; idx++){
			char_tanh_conv[idx].forward(cg, &(char_window._outputs[idx]));
		}
		char_left_lstm.forward(cg, getPNodes(char_tanh_conv, char_size));
		char_right_lstm.forward(cg, getPNodes(char_tanh_conv, char_size));
	}

};

#endif /* SRC_GlobalNodes_H_ */
