/*
 * AtomFeatures.h
 *
 *  Created on: Aug 25, 2016
 *      Author: mszhang
 */

#ifndef SRC_AtomFeatures_H_
#define SRC_AtomFeatures_H_

struct AtomFeatures {
public:
	string str_C0;
	string str_1C;  //equals str_1Wcn
	string str_2C; 

	string str_CT0;
	string str_1CT;  //equals str_1Wcn
	string str_2CT;

	string str_1W;
	string str_1WD;
	string str_1WL;
	int int_1WL;
	string str_1Wc0;
	vector<string> str_1Wci;

	string str_2W;
	string str_2WL;
	string str_2Wc0;
	string str_2Wcn;

public:
	void clear(){
		str_C0 = "";
		str_1C = "";
		str_2C = "";

		str_CT0 = "";
		str_1CT = "";
		str_2CT = "";

		str_1W = "";
		str_1WD = "";
		str_1WL = "";
		int_1WL = 0;
		str_1Wc0 = "";
		str_1Wci.clear();

		str_2W = "";
		str_2WL = "";
		str_2Wc0 = "";
		str_2Wcn = "";		
	}

};

#endif /* SRC_AtomFeatures_H_ */
