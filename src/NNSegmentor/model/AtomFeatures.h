/*
 * AtomFeatures.h
 *
 *  Created on: Aug 25, 2016
 *      Author: mszhang
 */

#ifndef SRC_AtomFeatures_H_
#define SRC_AtomFeatures_H_
#include "ModelParams.h"
struct AtomFeatures {
public:
	string str_C0;
	string str_1C;  //equals str_1Wcn
	string str_2C; 

	string str_CT0;
	string str_1CT;  //equals str_1Wcn
	string str_2CT;

	string str_1W;
	string str_1Wc0;
	vector<string> str_1Wci;

	string str_2W;
	string str_2Wc0;
	string str_2Wcn;

public:
	int sid_1WD;
	int sid_1WL;
	int sid_2WL;

public:
	int sid_C0;
	int sid_1C;  //equals str_1Wcn
	int sid_2C;

	int sid_CT0;
	int sid_1CT;  //equals str_1Wcn
	int sid_2CT;

	int sid_1W;
	int sid_1Wc0;
	vector<int> sid_1Wci;

	int sid_2W;
	int sid_2Wc0;
	int sid_2Wcn;

//no need to convert to sid;
public:
	string str_1AC;
	string str_2AC;
	int next_position;
public:
	IncLSTM1Builder* p_word_lstm;
	IncLSTM1Builder* p_action_lstm;
	LSTM1Builder* p_char_left_lstm;
	LSTM1Builder* p_char_right_lstm;
public:
	void clear(){
		str_C0 = "";
		str_1C = "";
		str_2C = "";

		str_CT0 = "";
		str_1CT = "";
		str_2CT = "";

		str_1W = "";
		str_1Wc0 = "";
		str_1Wci.clear();

		str_2W = "";
		str_2Wc0 = "";
		str_2Wcn = "";		

		sid_C0 = -1;
		sid_1C = -1;
		sid_2C = -1;

		//ids
		sid_CT0 = -1;
		sid_1CT = -1;
		sid_2CT = -1;

		sid_1W = -1;
		sid_1WD = -1;
		sid_1WL = -1;
		sid_1Wc0 = -1;
		sid_1Wci.clear();

		sid_2W = -1;
		sid_2WL = -1;
		sid_2Wc0 = -1;
		sid_2Wcn = -1;

		str_1AC = "";
		str_2AC = "";
		next_position = -1;
		p_word_lstm = NULL;
		p_action_lstm = NULL;
		p_char_left_lstm = NULL;
		p_char_right_lstm = NULL;
	}

public:
	void convert2Id(ModelParams* model){
		sid_C0 = model->chars.from_string(str_C0);
		sid_1C = model->chars.from_string(str_1C);
		sid_2C = model->chars.from_string(str_2C);

		sid_CT0 = model->charTypes.from_string(str_CT0);
		sid_1CT = model->charTypes.from_string(str_1CT);
		sid_2CT = model->charTypes.from_string(str_2CT);

		sid_1W = model->words.from_string(str_1W);
		sid_1Wc0 = model->chars.from_string(str_1Wc0);
		sid_1Wci.resize(str_1Wci.size());
		for (int idx = 0; idx < str_1Wci.size(); idx++){
			sid_1Wci[idx] = model->chars.from_string(str_1Wci[idx]);
		}

		sid_2W = model->words.from_string(str_2W);
		sid_2Wc0 = model->chars.from_string(str_2Wc0);
		sid_2Wcn = model->chars.from_string(str_2Wcn);
	}



};

#endif /* SRC_AtomFeatures_H_ */
