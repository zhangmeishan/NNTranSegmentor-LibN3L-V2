/*
 * Segmentor.cpp
 *
 *  Created on: Jan 25, 2016
 *      Author: mszhang
 */

#include "APSegmentor.h"

#include "Argument_helper.h"

Segmentor::Segmentor() {
	// TODO Auto-generated constructor stub
	srand(0);
	//Node::id = 0;
}

Segmentor::~Segmentor() {
	// TODO Auto-generated destructor stub
}

// all linear features are extracted from positive examples
int Segmentor::createAlphabet(const vector<Instance>& vecInsts) {
	cout << "Creating Alphabet..." << endl;

	int numInstance = vecInsts.size();

	unordered_map<string, int> word_stat;
	unordered_map<string, int> char_stat;

	assert(numInstance > 0);
	int count = 0;
	for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
		const Instance &instance = vecInsts[numInstance];
		for (int idx = 0; idx < instance.wordsize(); idx++) {
			word_stat[normalize_to_lowerwithdigit(instance.words[idx])]++;
		}
		for (int idx = 0; idx < instance.charsize(); idx++) {
			char_stat[instance.chars[idx]]++;
		}
		count += instance.wordsize();
	}
	word_stat[nullkey] = m_options.wordCutOff;
	char_stat[nullkey] = m_options.charCutOff;
	m_driver._modelparams.words.initial(word_stat, m_options.wordCutOff);
	m_driver._modelparams.chars.initial(char_stat, m_options.charCutOff);

	static unordered_map<string, int>::const_iterator elem_iter;
	for (elem_iter = word_stat.begin(); elem_iter != word_stat.end(); elem_iter++){
		if (elem_iter->second > count / 50000 + 3){
			m_driver._hyperparams.dicts.insert(elem_iter->first);
		}
	}

	m_driver._modelparams.actions.clear();
	m_driver._modelparams.wordLengths.clear();
	m_driver._modelparams.dictionarys.clear();
	m_driver._modelparams.charTypes.clear();

	vector<CStateItem> state(m_driver._hyperparams.maxlength + 1);
	vector<string> output;
	CAction answer;
	Metric eval;
	int actionNum;
	eval.reset();
	for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
		const Instance &instance = vecInsts[numInstance];
		actionNum = 0;
		state[actionNum].clear();
		state[actionNum].setInput(&instance.chars);
		while (!state[actionNum].IsTerminated()) {
			state[actionNum].getGoldAction(instance.words, answer);
			m_driver._modelparams.actions.from_string(answer.str());

			state[actionNum].prepare(m_driver._hyperparams.dicts, NULL);
			m_driver._modelparams.wordLengths.from_string(state[actionNum]._atomFeat.str_1WL);
			m_driver._modelparams.dictionarys.from_string(state[actionNum]._atomFeat.str_1WD);
			m_driver._modelparams.charTypes.from_string(state[actionNum]._atomFeat.str_CT0);

			state[actionNum].move(&(state[actionNum + 1]), answer);
			actionNum++;
		}

		if (actionNum - 1 != instance.charsize()) {
			std::cout << "action number is not correct, please check" << std::endl;
		}
		state[actionNum].getSegResults(output);

		instance.evaluate(output, eval);

		if (!eval.bIdentical()) {
			std::cout << "error state conversion!" << std::endl;
			exit(0);
		}

		if ((numInstance + 1) % m_options.verboseIter == 0) {
			cout << numInstance + 1 << " ";
			if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
				cout << std::endl;
			cout.flush();
		}
		if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
			break;
	}


	cout << numInstance << " " << endl;
	cout << "Action num: " << m_driver._modelparams.actions.size() << endl;
	cout << "Total word num: " << word_stat.size() << endl;
	cout << "Total char num: " << char_stat.size() << endl;

	cout << "Remain word num: " << m_driver._modelparams.words.size() << endl;
	cout << "Remain char num: " << m_driver._modelparams.chars.size() << endl;
	cout << "Remain charType num: " << m_driver._modelparams.charTypes.size() << endl;
	cout << "Remain wordLength num: " << m_driver._modelparams.wordLengths.size() << endl;
	cout << "Remain dictionary type  num: " << m_driver._modelparams.dictionarys.size() << endl;


	cout << "Dictionary word num: " << m_driver._hyperparams.dicts.size() << endl;

	m_driver._modelparams.actions.set_fixed_flag(true);
	m_driver._modelparams.wordLengths.set_fixed_flag(true);
	m_driver._modelparams.dictionarys.set_fixed_flag(true);
	m_driver._modelparams.charTypes.set_fixed_flag(true);

	return 0;
}

void Segmentor::getGoldActions(const vector<Instance>& vecInsts, vector<vector<CAction> >& vecActions){
	vecActions.clear();

	Metric eval;
	vector<CStateItem> state(m_driver._hyperparams.maxlength + 1);
	vector<string> output;
	CAction answer;
	eval.reset();
	static int numInstance, actionNum;
	vecActions.resize(vecInsts.size());
	for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
		const Instance &instance = vecInsts[numInstance];

		actionNum = 0;
		state[actionNum].clear();
		state[actionNum].setInput(&instance.chars);
		while (!state[actionNum].IsTerminated()) {
			state[actionNum].getGoldAction(instance.words, answer);
			vecActions[numInstance].push_back(answer);
			state[actionNum].move(&state[actionNum + 1], answer);
			actionNum++;
		}

		if (actionNum - 1 != instance.charsize()) {
			std::cout << "action number is not correct, please check" << std::endl;
		}
		state[actionNum].getSegResults(output);

		instance.evaluate(output, eval);

		if (!eval.bIdentical()) {
			std::cout << "error state conversion!" << std::endl;
			exit(0);
		}

		if ((numInstance + 1) % m_options.verboseIter == 0) {
			cout << numInstance + 1 << " ";
			if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
				cout << std::endl;
			cout.flush();
		}
		if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
			break;
	}
}

void Segmentor::train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile) {
	if (optionFile != "")
		m_options.load(optionFile);

	m_options.showOptions();
	vector<Instance> trainInsts, devInsts, testInsts;
	m_pipe.readInstances(trainFile, trainInsts, m_driver._hyperparams.maxlength, m_options.maxInstance);
	if (devFile != "")
		m_pipe.readInstances(devFile, devInsts, m_driver._hyperparams.maxlength, m_options.maxInstance);
	if (testFile != "")
		m_pipe.readInstances(testFile, testInsts, m_driver._hyperparams.maxlength, m_options.maxInstance);

	vector<vector<Instance> > otherInsts(m_options.testFiles.size());
	for (int idx = 0; idx < m_options.testFiles.size(); idx++) {
		m_pipe.readInstances(m_options.testFiles[idx], otherInsts[idx], m_driver._hyperparams.maxlength, m_options.maxInstance);
	}

	createAlphabet(trainInsts);
	m_driver._hyperparams.setRequared(m_options);
	m_driver.initial();

	vector<vector<CAction> > trainInstGoldactions;
	getGoldActions(trainInsts, trainInstGoldactions);
	double bestFmeasure = 0;

	int inputSize = trainInsts.size();

	std::vector<int> indexes;
	for (int i = 0; i < inputSize; ++i)
		indexes.push_back(i);

	static Metric eval, metric_dev, metric_test;

	int maxIter = m_options.maxIter * (inputSize / m_options.batchSize + 1);
	int oneIterMaxRound = (inputSize + m_options.batchSize - 1) / m_options.batchSize;
	std::cout << "maxIter = " << maxIter << std::endl;
	int devNum = devInsts.size(), testNum = testInsts.size();

	static vector<vector<string> > decodeInstResults;
	static vector<string> curDecodeInst;
	static bool bCurIterBetter;
	static vector<vector<string> > subInstances;
	static vector<vector<CAction> > subInstGoldActions;

	std::cout << "Collect gold-standard features..." << std::endl;
	for (int idx = 0; idx < inputSize; idx++){
		subInstances.clear();
		subInstGoldActions.clear();
		subInstances.push_back(trainInsts[idx].chars);
		subInstGoldActions.push_back(trainInstGoldactions[idx]);
		m_driver.extractFeat(subInstances, subInstGoldActions);
		if ((idx + 1) % m_options.verboseIter == 0) {
			cout << idx + 1 << " ";
			if ((idx + 1) % (40 * m_options.verboseIter) == 0)
				cout << std::endl;
			cout.flush();
		}
	}
	cout << inputSize << std::endl;
	m_driver._modelparams.setFixed(m_options.base);

	for (int iter = 0; iter < maxIter; ++iter) {
		std::cout << "##### Iteration " << iter << std::endl;
		srand(iter);
		random_shuffle(indexes.begin(), indexes.end());
		std::cout << "random: " << indexes[0] << ", " << indexes[indexes.size() - 1] << std::endl;
		bool bEvaluate = false;

		if (m_options.batchSize == 1){
			eval.reset();
			bEvaluate = true;
			for (int idy = 0; idy < inputSize; idy++) {
				subInstances.clear();
				subInstGoldActions.clear();
				subInstances.push_back(trainInsts[indexes[idy]].chars);
				subInstGoldActions.push_back(trainInstGoldactions[indexes[idy]]);

				double cost = m_driver.train(subInstances, subInstGoldActions);

				eval.overall_label_count += m_driver._eval.overall_label_count;
				eval.correct_label_count += m_driver._eval.correct_label_count;

				if ((idy + 1) % (m_options.verboseIter * 10) == 0) {
					std::cout << "current: " << idy + 1 << ", Cost = " << cost << ", Correct(%) = " << eval.getAccuracy() << std::endl;
				}
				m_driver.updateModel();
			}
			std::cout << "current: " << iter + 1 << ", Correct(%) = " << eval.getAccuracy() << std::endl;
		}
		else{
			if (iter == 0)eval.reset();
			subInstances.clear();
			subInstGoldActions.clear();
			for (int idy = 0; idy < m_options.batchSize; idy++) {
				subInstances.push_back(trainInsts[indexes[idy]].chars);
				subInstGoldActions.push_back(trainInstGoldactions[indexes[idy]]);
			}
			double cost = m_driver.train(subInstances, subInstGoldActions);

			eval.overall_label_count += m_driver._eval.overall_label_count;
			eval.correct_label_count += m_driver._eval.correct_label_count;

			if ((iter + 1) % (m_options.verboseIter) == 0) {
				std::cout << "current: " << iter + 1 << ", Cost = " << cost << ", Correct(%) = " << eval.getAccuracy() << std::endl;
				eval.reset();
				bEvaluate = true;
			}

			m_driver.updateModel();
		}

		if (bEvaluate && devNum > 0) {
			bCurIterBetter = false;
			if (!m_options.outBest.empty())
				decodeInstResults.clear();
			metric_dev.reset();
			for (int idx = 0; idx < devInsts.size(); idx++) {
				predict(devInsts[idx], curDecodeInst);
				devInsts[idx].evaluate(curDecodeInst, metric_dev);
				if (!m_options.outBest.empty()) {
					decodeInstResults.push_back(curDecodeInst);
				}
			}
			std::cout << "dev:" << std::endl;
			metric_dev.print();

			if (!m_options.outBest.empty() && metric_dev.getAccuracy() > bestFmeasure) {
				m_pipe.outputAllInstances(devFile + m_options.outBest, decodeInstResults);
				bCurIterBetter = true;
			}


			if (testNum > 0) {
				if (!m_options.outBest.empty())
					decodeInstResults.clear();
				metric_test.reset();
				for (int idx = 0; idx < testInsts.size(); idx++) {
					predict(testInsts[idx], curDecodeInst);
					testInsts[idx].evaluate(curDecodeInst, metric_test);
					if (bCurIterBetter && !m_options.outBest.empty()) {
						decodeInstResults.push_back(curDecodeInst);
					}
				}
				std::cout << "test:" << std::endl;
				metric_test.print();

				if (!m_options.outBest.empty() && bCurIterBetter) {
					m_pipe.outputAllInstances(testFile + m_options.outBest, decodeInstResults);
				}
			}

			for (int idx = 0; idx < otherInsts.size(); idx++) {
				std::cout << "processing " << m_options.testFiles[idx] << std::endl;
				if (!m_options.outBest.empty())
					decodeInstResults.clear();
				metric_test.reset();
				for (int idy = 0; idy < otherInsts[idx].size(); idy++) {
					predict(otherInsts[idx][idy], curDecodeInst);
					otherInsts[idx][idy].evaluate(curDecodeInst, metric_test);
					if (bCurIterBetter && !m_options.outBest.empty()) {
						decodeInstResults.push_back(curDecodeInst);
					}
				}
				std::cout << "test:" << std::endl;
				metric_test.print();

				if (!m_options.outBest.empty() && bCurIterBetter) {
					m_pipe.outputAllInstances(m_options.testFiles[idx] + m_options.outBest, decodeInstResults);
				}
			}


			if (m_options.saveIntermediate && metric_dev.getAccuracy() > bestFmeasure) {
				std::cout << "Exceeds best previous DIS of " << bestFmeasure << ". Saving model file.." << std::endl;
				bestFmeasure = metric_dev.getAccuracy();
				writeModelFile(modelFile);
			}
		}
	}
}

void Segmentor::predict(const Instance& input, vector<string>& output) {
	m_driver.decode(input.chars, output);
}

void Segmentor::test(const string& testFile, const string& outputFile, const string& modelFile) {
	loadModelFile(modelFile);
	vector<Instance> testInsts;
	m_pipe.readInstances(testFile, testInsts, m_options.maxInstance);

	vector<vector<string> > testInstResults(testInsts.size());
	Metric metric_test;
	metric_test.reset();
	for (int idx = 0; idx < testInsts.size(); idx++) {
		vector<string> result_labels;
		predict(testInsts[idx], testInstResults[idx]);
		testInsts[idx].evaluate(testInstResults[idx], metric_test);
	}
	std::cout << "test:" << std::endl;
	metric_test.print();

	std::ofstream os(outputFile.c_str());

	for (int idx = 0; idx < testInsts.size(); idx++) {
		for (int idy = 0; idy < testInstResults[idx].size(); idy++){
			os << testInstResults[idx][idy] << " ";
		}
		os << std::endl;
	}
	os.close();
}


void Segmentor::loadModelFile(const string& inputModelFile) {

}

void Segmentor::writeModelFile(const string& outputModelFile) {

}

int main(int argc, char* argv[]) {
	std::string trainFile = "", devFile = "", testFile = "", modelFile = "";
	std::string wordEmbFile = "", optionFile = "";
	std::string outputFile = "";
	bool bTrain = false;
	dsr::Argument_helper ah;

	ah.new_flag("l", "learn", "train or test", bTrain);
	ah.new_named_string("train", "trainCorpus", "named_string", "training corpus to train a model, must when training", trainFile);
	ah.new_named_string("dev", "devCorpus", "named_string", "development corpus to train a model, optional when training", devFile);
	ah.new_named_string("test", "testCorpus", "named_string",
		"testing corpus to train a model or input file to test a model, optional when training and must when testing", testFile);
	ah.new_named_string("model", "modelFile", "named_string", "model file, must when training and testing", modelFile);
	ah.new_named_string("word", "wordEmbFile", "named_string", "pretrained word embedding file to train a model, optional when training", wordEmbFile);
	ah.new_named_string("option", "optionFile", "named_string", "option file to train a model, optional when training", optionFile);
	ah.new_named_string("output", "outputFile", "named_string", "output file to test, must when testing", outputFile);

	ah.process(argc, argv);

	Segmentor segmentor;
	if (bTrain) {
		segmentor.train(trainFile, devFile, testFile, modelFile, optionFile);
	}
	else {
		segmentor.test(testFile, outputFile, modelFile);
	}

	//test(argv);
	//ah.write_values(std::cout);

}
