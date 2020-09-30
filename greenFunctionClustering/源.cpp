#include <omp.h>
#include <queue>
#include <vector>
#include <time.h>
#include<numeric>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include "kdtree.h"
#include <stdlib.h>
#include <iostream>
#include <limits.h>
#include <algorithm>
#include <Eigen/SparseCholesky>


using namespace std;
using namespace kdt;
using namespace Eigen;

class KDTreePoint
{
private:
	const double* pt_;
public:
	static int DIM;
	KDTreePoint(const double* pt) :pt_(pt)
	{
	}

	const double& operator[](const int index) const
	{
		assert(index < DIM);
		return pt_[index];
	}
};
int KDTreePoint::DIM = 3;


vector<vector<double>> LoadDataSet(string fileName);
SparseMatrix<int> GetknnGraphKDTree(const vector<vector<double>> &dataMat, int kNearestNeighborNum);
SparseMatrix<int> GetknnGraph(const vector<vector<double>> &dataMat, int kNearestNeighborNum);
SparseMatrix<int> GetknnGraph(const vector<vector<double>> &dataMat, int kNearestNeighborNum, const vector<int> &samples);
vector<vector<pair<unsigned int, unsigned short>>> GetknnGraph_(const vector<vector<double>> &, int);
void ConstructHeap(vector<pair<double, int>> &a, int n, pair<double, int> value);
void UpdataHeap(vector<pair<double, int>> &a);
vector<int> argTopK(vector<double> arr, int k);
int getsnnNum(const SparseMatrix<int> &knnGraph, int i, int j);
bool cmp(pair<double, pair<int, int>> a, pair<double, pair<int, int>> b);
pair<SparseMatrix<int>, SparseMatrix<double>> GetSNNDistLaplacianMat(const vector<vector<double>> &, const SparseMatrix<int> &, int, int);
vector<vector<double> > GetGreenFuncGrad(const SparseMatrix<int> &, const SparseMatrix<double> &, int);
vector<vector<double> > kMeansPlusPlusInit(const vector<vector<double> > &data, int clusterNum);
vector<int> kMeans(const vector<vector<double> > &data, int clusterNum, int initNum);


int main()
{
	//vector<Triplet<int>> coefficients;
	//coefficients.push_back(Triplet<int>(0, 0, 1));
	//coefficients.push_back(Triplet<int>(0, 2, 1));
	//coefficients.push_back(Triplet<int>(1, 1, 1));
	//coefficients.push_back(Triplet<int>(2, 0, 1));
	//coefficients.push_back(Triplet<int>(2, 2, 2));
	//SparseMatrix<int> knnGraph(3, 3);
	//knnGraph.setFromTriplets(coefficients.begin(), coefficients.end());
	//cout << knnGraph.toDense() << endl;
	//SparseMatrix<double> L;
	//L = knnGraph.cast<double> ();
	//cout << L.toDense() << endl;
	//knnGraph.coeffRef(2, 2) = 10;
	//cout << knnGraph.toDense() << endl;
	//cout << L.toDense() << endl;
	//SimplicialLDLT<SparseMatrix<double> > solver;
	//solver.compute(L);
	//cout << solver.matrixL() << endl;
	//cout << solver.matrixU() << endl;
	//cout << solver.vectorD() << endl;
	//VectorXd b(3);
	//b(0) = 0.5;
	//b(1) = 1;
	//b(2) = 1;
	//VectorXd g = solver.solve(b);
	///*cout << knnGraph.toDense() << endl;
	//cout << knnGraph.nonZeros() << endl;*/
	//cout << b << endl;
	//cout << g << endl;
	
	/*vector<vector<double>> dataMat = LoadDataSet("complexStructure.txt");
	vector<int> labels;
	labels = kMeans(dataMat, 3, 5);
	FILE *fp;
	errno_t err;
	if (err = fopen_s(&fp, "labels.txt", "w"))
	{
		printf("timeFile error value: %d", err);
		exit(1);
	}
	for (int i = 0; i < labels.size(); ++i)
	{
			fprintf(fp, "%d\n", labels[i]);
	}
	fclose(fp);*/

	/*vector<pair<double, pair<int, int>>> a(8);
	a[0] = make_pair(0.3, make_pair(0, 1));
	a[1] = make_pair(0.8, make_pair(2, 3));
	a[2] = make_pair(0.9, make_pair(2, 1));
	a[3] = make_pair(0.2, make_pair(1, 2));
	a[4] = make_pair(0.4, make_pair(1, 3));
	a[5] = make_pair(0.5, make_pair(1, 0));
	a[6] = make_pair(0.6, make_pair(3, 0));
	a[7] = make_pair(0.7, make_pair(3, 2));
	for (int i = 0; i < a.size(); ++i)
		cout << a[i].second.first << "\t" << a[i].second.second << "\t" << a[i].first << endl;
	sort(a.begin(), a.end(), cmp);
	for (int i = 0; i < a.size(); ++i)
		cout << a[i].second.first << "\t" << a[i].second.second << "\t" << a[i].first << endl;*/

	/*int a = 5;
	unsigned int b = a;
	unsigned short c = 1;
	vector<vector<pair<unsigned int, unsigned short>>> knnGraph(10000);
	for (int i = 0; i < knnGraph.size(); ++i)
		for (int j = 0; j < (int)i * 0.2; ++j)
			knnGraph[i].push_back(make_pair(b, c));
	int s = 0;
	for (int i = 0; i < knnGraph.size(); ++i)
		s += knnGraph[i].size();
	cout << s << endl;*/

	/*vector<vector<double>> dataMat;
	dataMat = LoadDataSet("reuters10kData.txt");
	vector<vector<pair<unsigned int, unsigned short>>> knnGraph;
	knnGraph = GetknnGraph_(dataMat, 1000);*/
	/*int s = 0;
	vector<int> len(100, 0);
	for (int i = 0; i < knnGraph.size(); ++i)
		for (int j = 0; j < knnGraph[i].size(); ++j)
			if (knnGraph[i][j].second < 100)
				++len[knnGraph[i][j].second];
	cout << len[0] << endl;
	cout << len[1] << endl;
	cout << len[2] << endl;
	cout << len[3] << endl;*/
	/*SparseMatrix<int> knnGraph;
	knnGraph = GetknnGraph(dataMat, 1000);
	cout << knnGraph.nonZeros() << endl;*/
	
	//int kNearestNeighborNum, sharedNearestNeighborNum, clusterNum;
	//clock_t t1, t2, t3, t4;
	//string fileName;
	//vector<vector<double>> dataMat;
	//SparseMatrix<int> knnGraph;
	//SparseMatrix<int> L;
	//SparseMatrix<double> distGraph;
	//vector<vector<double> > grad;
	//vector<int> labels;
	//srand((unsigned int)time(NULL));

	//cout << "file name(no more than 50 characters):";
	//cin >> fileName;
	//cout << "k-nearest neighbor:";
	//cin >> kNearestNeighborNum;
	//cout << "shared nearest-neighbor:";
	//cin >> sharedNearestNeighborNum;
	//cout << "the number of clusters:";
	//cin >> clusterNum;
	//dataMat = LoadDataSet(fileName);
	////t1 = clock();
	//vector<int> samples;
	//while (samples.size() < 10000)
	//{
	//	int id = int(((double)rand() / RAND_MAX)*dataMat.size());
	//	if (find(samples.begin(), samples.end(), id) == samples.end())
	//		samples.push_back(id);
	//}
	//knnGraph = GetknnGraph(dataMat, sharedNearestNeighborNum, samples);
	////t2 = clock();
	////cout << (double)(t2 - t1) / CLOCKS_PER_SEC << endl;
	//tie(L, distGraph) = GetSNNDistLaplacianMat(dataMat, knnGraph, kNearestNeighborNum, sharedNearestNeighborNum);
	//grad = GetGreenFuncGrad(L, distGraph, min((int)(pow(kNearestNeighborNum, 1) * pow(clusterNum, 2)), (int)((L.nonZeros() - dataMat.size()) / 2)));
	//labels = kMeans(grad, clusterNum, 100);
	//FILE *fp;
	//errno_t err;
	//if (err = fopen_s(&fp, "labels.txt", "w"))
	//{
	//	printf("timeFile error value: %d", err);
	//	exit(1);
	//}
	//for (int i = 0; i < labels.size(); ++i)
	//{
	//	fprintf(fp, "%d\n", labels[i]);
	//}
	//fclose(fp);

	/*vector<double *> data;
	data.resize(5);
	for (int i = 0; i < 5; ++i)
		data[i] = (double *)malloc(3 * sizeof(double));
	for (int i = 0; i < 5; ++i)
		for (int j = 0; j < 3; ++j)
			data[i][j] = i;
	for (int i = 0; i < 5; ++i)
	{
		for (int j = 0; j < 3; ++j)
			cout << data[i][j] << ' ';
		cout << endl;
	}
	vector<KDTreePoint> points(data.begin(), data.end());
	KDTree<KDTreePoint> kdtree(points);
	KDTreePoint::DIM = 3;
	double *b = (double *)malloc(3 * sizeof(double));
	for (int i = 0; i < 3; ++i)
		b[i] = 0;
	vector<int> ids = kdtree.knnSearch(b, 3);
	cout << ids.size() << endl;
	for (int i = 0; i < 3; ++i)
	{
		cout << ids[i] << endl;
	}*/

	/*vector<vector<double>> data;
	for (int i = 0; i < 5; ++i)
	{
		vector<double> temp(3, i);
		data.push_back(temp);
		for (int j = 0; j <= i; ++j)
			cout << data[j].data() << " ";
		cout << endl;
	}
	vector<double *> data0;
	for (int i = 0; i < data.size(); ++i)
		data0.push_back(data[i].data());
	for (int i = 0; i < 5; ++i)
	{
		for (int j = 0; j < 3; ++j)
			cout << data0[i][j] << ' ';
		cout << endl;
	}
	vector<KDTreePoint> points(data0.begin(), data0.end());
	KDTree<KDTreePoint> kdtree(points);
	KDTreePoint::DIM = 3;
	vector<int> ids = kdtree.knnSearch(KDTreePoint(data0[0]), 3);
	cout << ids.size() << endl;
	for (int i = 0; i < 3; ++i)
	{
		cout << ids[i] << endl;
	}
	cout << typeid(data0).name();*/
	getchar();
	return 0;
}

vector<vector<double>> LoadDataSet(string fileName)
{
	vector<vector<double>> dataMat;
	ifstream fp(fileName);
	while (!fp.eof())
	{
		string line;
		getline(fp, line);
		if (!line.empty())
		{
			istringstream input(line);
			vector<double> tempVector;
			double temp;
			while (input >> temp)
				tempVector.push_back(temp);
			dataMat.push_back(tempVector);
		}
	}
	fp.close();
	return dataMat;
}

SparseMatrix<int> GetknnGraph(const vector<vector<double>> &dataMat, int kNearestNeighborNum)
{
	int dataPointNum = dataMat.size(), dataDimension = dataMat[0].size();
	vector<Triplet<int>> coefficients;
	vector<vector<Triplet<int> > > buffers;
	SparseMatrix<int> knnGraph(dataPointNum, dataPointNum);
	int threadNum = omp_get_max_threads();
	buffers.resize(threadNum);
	cout << "knnGraph" << endl;
#pragma omp parallel for
	for (int i = 0; i < dataPointNum; ++i)
	{
		auto id = omp_get_thread_num();
		vector<double> distI(dataPointNum, 0);
		vector<int> index(kNearestNeighborNum + 1);
		for (int j = 0; j < dataPointNum; ++j)
		{
			for (int k = 0; k < dataDimension; ++k)
				distI[j] += pow(dataMat[i][k] - dataMat[j][k], 2);
		}
		index = argTopK(distI, kNearestNeighborNum + 1);
		for (int j = 0; j <= kNearestNeighborNum; ++j)
			buffers[id].push_back(Triplet<int>(index[j], i, 1));
	}

	for (auto & buffer : buffers)
	{
		move(buffer.begin(), buffer.end(), back_inserter(coefficients));
		buffer.clear();
		vector<Triplet<int> >().swap(buffer);
	}
	knnGraph.setFromTriplets(coefficients.begin(), coefficients.end());
	knnGraph.diagonal() = VectorXi::Constant(dataPointNum, 0);
	knnGraph.prune(1, 0);
	return knnGraph;
}

SparseMatrix<int> GetknnGraph(const vector<vector<double>> &dataMat, int kNearestNeighborNum, const vector<int> &samples)
{
	int dataPointNum = dataMat.size(), dataDimension = dataMat[0].size(), sampleNum = samples.size();
	vector<Triplet<int>> coefficients;
	vector<vector<Triplet<int> > > buffers;
	SparseMatrix<int> knnGraph(sampleNum, dataPointNum);
	int threadNum = omp_get_max_threads();
	buffers.resize(threadNum);
	cout << "knnGraph" << endl;
#pragma omp parallel for
	for (int i = 0; i < dataPointNum; ++i)
	{
		if (i % 200 == 0)
			cout << i << endl;
		auto id = omp_get_thread_num();
		vector<double> distI(sampleNum, 0);
		vector<int> index(kNearestNeighborNum + 1);
		for (int j = 0; j < sampleNum; ++j)
		{
			for (int k = 0; k < dataDimension; ++k)
				distI[j] += pow(dataMat[i][k] - dataMat[samples[j]][k], 2);
		}
		index = argTopK(distI, kNearestNeighborNum + 1);
		for (int j = 0; j <= kNearestNeighborNum; ++j)
			buffers[id].push_back(Triplet<int>(index[j], i, 1));
	}

	for (auto & buffer : buffers)
	{
		move(buffer.begin(), buffer.end(), back_inserter(coefficients));
		buffer.clear();
		vector<Triplet<int> >().swap(buffer);
	}
	knnGraph.setFromTriplets(coefficients.begin(), coefficients.end());
	//knnGraph.diagonal() = VectorXi::Constant(dataPointNum, 0);
	knnGraph.prune(1, 0);
	return knnGraph;
}

vector<vector<pair<unsigned int, unsigned short>>> GetknnGraph_(const vector<vector<double>> &dataMat, int kNearestNeighborNum)
{
	int dataPointNum = dataMat.size(), dataDimension = dataMat[0].size();
	vector<vector<pair<unsigned int, unsigned short>>> knnGraph(dataPointNum);
	cout << "knnGraph" << endl;
#pragma omp parallel for
	for (int i = 0; i < dataPointNum; ++i)
	{
		if (i % 500 == 0)
			cout << i << endl;
		vector<double> distI(dataPointNum, 0);
		vector<int> index(kNearestNeighborNum + 1);
		for (int j = 0; j < dataPointNum; ++j)
		{
			for (int k = 0; k < dataDimension; ++k)
				distI[j] += pow(dataMat[i][k] - dataMat[j][k], 2);
		}
		index = argTopK(distI, kNearestNeighborNum + 1);
		sort(index.begin(), index.end());
		for (vector<int>::iterator it = index.begin(); it != index.end();)
			if (*it == i)
			{
				it = index.erase(it);
				break;
			}
			else
				++it;
		unsigned short count = 1;
		unsigned int start = index[0];
		for (int j = 1; j < index.size(); ++j)
		{
			if (index[j - 1] + 1 == index[j])
				++count;
			else
			{
				knnGraph[i].push_back(make_pair(start, count));
				start = index[j];
				count = 1;
			}
		}
		knnGraph[i].push_back(make_pair(start, count));
	}
	return knnGraph;
}

SparseMatrix<int> GetknnGraphKDTree(const vector<vector<double>> &dataMat, int kNearestNeighborNum)
{
	int dataPointNum = dataMat.size(), dataDimension = dataMat[0].size();
	vector<const double *>dataPointer;
	vector<Triplet<int>> coefficients;
	vector<vector<Triplet<int> > > buffers;
	SparseMatrix<int> knnGraph(dataPointNum, dataPointNum);
	for (int i = 0; i < dataPointNum; ++i)
		dataPointer.push_back(dataMat[i].data());
	vector<KDTreePoint> points(dataPointer.begin(), dataPointer.end());
	KDTree<KDTreePoint> kdtree(points);
	int threadNum = omp_get_max_threads();
	KDTreePoint::DIM = dataDimension;
	buffers.resize(threadNum);
	cout << "knnGraph" << endl;
#pragma omp parallel for
	for (int i = 0; i < dataPointNum; ++i)
	{
		/*if (i % 1000 == 0)
			printf("L1: %d\n", i);*/
		KDTreePoint p(dataPointer[i]);
		vector<int> knnIndex = kdtree.knnSearch(p, kNearestNeighborNum + 1);
		auto id = omp_get_thread_num();
		for (size_t j = 0; j < knnIndex.size(); ++j)
			buffers[id].push_back(Triplet<int>(knnIndex[j], i, 1));
	}

	for (auto & buffer : buffers)
	{
		move(buffer.begin(), buffer.end(), back_inserter(coefficients));
		buffer.clear();
		vector<Triplet<int> >().swap(buffer);
	}
	knnGraph.setFromTriplets(coefficients.begin(), coefficients.end());
	knnGraph.diagonal() = VectorXi::Constant(dataPointNum, 0);
	knnGraph.prune(1, 0);
	return knnGraph;
}

void ConstructHeap(vector<pair<double, int>> &a, int n, pair<double, int> value)
{
	a[n] = value;
	int j = (n - 1) / 2;
	pair<double, int> temp = value;
	while (j >= 0 && n != 0)
	{
		if (a[j].first > temp.first)
			break;
		a[n] = a[j];
		n = j;
		j = (n - 1) / 2;
	}
	a[n] = temp;
}

void UpdataHeap(vector<pair<double, int>> &a)
{
	int i = 0, j = 1, n = a.size();
	pair<double, int> temp = a[0];
	while (j < n)
	{
		if (j + 1 < n && a[j + 1].first >= a[j].first)
			++j;
		if (a[j].first <= temp.first)
			break;
		a[i] = a[j];
		i = j;
		j = 2 * i + 1;
	}
	a[i] = temp;
}

vector<int> argTopK(vector<double> arr, int k)
{
	int n = arr.size();
	vector<pair<double, int>> heap(k);
	vector<int> index(k);
	for (int i = 0; i < n; ++i)
	{
		pair<double, int> temp(arr[i], i);
		if (i < k)
			ConstructHeap(heap, i, temp);
		else
		{
			if (heap[0].first > temp.first)
			{
				heap[0] = temp;
				UpdataHeap(heap);
			}
		}
	}
	for (int i = 0; i < k; ++i)
		index[i] = heap[i].second;
	return index;
}

int getsnnNum(const SparseMatrix<int> &knnGraph, int i, int j)
{
	int snnNum = 0;
	SparseMatrix<int>::InnerIterator itI(knnGraph, i);
	SparseMatrix<int>::InnerIterator itJ(knnGraph, j);
	while (itI && itJ)
	{
		//cout << itI.row() << " " << itJ.row() << endl;
		if (itI.row() == itJ.row())
		{
			++snnNum;
			++itI;
			++itJ;
		}
		else if (itI.row() < itJ.row())
			++itI;
		else
			++itJ;
	}
	return snnNum;
}

bool cmp(pair<double, pair<int, int>> a, pair<double, pair<int, int>> b)
{
	if (a.first != b.first)
		return a.first < b.first;
	else if (a.second.first != b.second.first)
		return a.second.first < b.second.first;
	else
		return a.second.second < b.second.second;
}

pair<SparseMatrix<int>, SparseMatrix<double>> GetSNNDistLaplacianMat(const vector<vector<double>> &dataMat,
	const SparseMatrix<int> &knnGraph, int kNearestNeighborNum, int sharedNearestNeighborNum)
{
	int dataPointNum = dataMat.size(), dataDimension = dataMat[0].size(), threadNum = omp_get_max_threads();

	vector<Triplet<int> > laplacianMatCoefficients;
	vector<vector<Triplet<int> > > laplacianMatBuffers(threadNum);
	SparseMatrix<int> L(dataPointNum, dataPointNum);

	vector<Triplet<double> > distGraphCoefficients;
	vector<vector<Triplet<double> > > distGraphBuffers(threadNum);
	SparseMatrix<double> distGraph(dataPointNum, dataPointNum);

	//mutual knn graph
	vector<vector<double> > dist(threadNum);
	vector<vector<int> > index(threadNum);
	vector<vector<int> > index2(threadNum);
	int knn2 = 15;
	for (int i = 0; i < threadNum; ++i)
	{
		dist[i].resize(dataPointNum);
		index[i].resize(kNearestNeighborNum + 1);
		index2[i].resize(knn2);
	}
	cout << "L1" << endl;
#pragma omp parallel for
	for (int i = 0; i < dataPointNum; ++i)
	{
		if (i % 200 == 0)
			cout << i << endl;
		auto threadID = omp_get_thread_num();
		for (int j = 0; j < dataPointNum; ++j)
		{
			/*int knnNum = 0;
			for (SparseMatrix<int>::InnerIterator it(knnGraph, i); it; ++it)
				if (knnGraph.coeff(it.row(), j) == 1)
					++knnNum;
			dist[threadID][j] = 1 - (double)knnNum / sharedNearestNeighborNum;*/
			dist[threadID][j] = 1 - (double)getsnnNum(knnGraph, i, j) / sharedNearestNeighborNum;
		}
		index[threadID] = argTopK(dist[threadID], kNearestNeighborNum + 1);
		index2[threadID] = argTopK(dist[threadID], knn2);
		for (int j = 0; j <= kNearestNeighborNum; ++j)
		{
			laplacianMatBuffers[threadID].push_back(Triplet<int>(i, index[threadID][j], 1));
			laplacianMatBuffers[threadID].push_back(Triplet<int>(index[threadID][j], i, 1));
			distGraphBuffers[threadID].push_back(Triplet<double>(i, index[threadID][j], dist[threadID][index[threadID][j]]));
			distGraphBuffers[threadID].push_back(Triplet<double>(index[threadID][j], i, dist[threadID][index[threadID][j]]));
		}
		for (int j = 0; j < knn2; ++j)
		{
			laplacianMatBuffers[threadID].push_back(Triplet<int>(i, index2[threadID][j], 1));
			laplacianMatBuffers[threadID].push_back(Triplet<int>(index2[threadID][j], i, 1));
			distGraphBuffers[threadID].push_back(Triplet<double>(i, index2[threadID][j], dist[threadID][index2[threadID][j]]));
			distGraphBuffers[threadID].push_back(Triplet<double>(index2[threadID][j], i, dist[threadID][index2[threadID][j]]));
		}
	}
	for (auto & buffer : laplacianMatBuffers)
	{
		move(buffer.begin(), buffer.end(), back_inserter(laplacianMatCoefficients));
		buffer.clear();
		vector<Triplet<int> >().swap(buffer);
	}
	for (auto & buffer : distGraphBuffers)
	{
		move(buffer.begin(), buffer.end(), back_inserter(distGraphCoefficients));
		buffer.clear();
		vector<Triplet<double> >().swap(buffer);
	}
	L.setFromTriplets(laplacianMatCoefficients.begin(), laplacianMatCoefficients.end());
	distGraph.setFromTriplets(distGraphCoefficients.begin(), distGraphCoefficients.end());

	//MST of all connected components
	int connectedComponentNum = 0, currentDataPointIndex = 0;
	vector<int> connectedComponentIndexOfData(dataPointNum, 0);
	vector<bool> visit(dataPointNum, false);
	vector<int> stack;
	cout << "connectedComponent" << endl;
	while (true)
	{
		while (currentDataPointIndex < dataPointNum && visit[currentDataPointIndex])
			++currentDataPointIndex;
		if (currentDataPointIndex == dataPointNum)
			break;
		visit[currentDataPointIndex] = true;
		connectedComponentIndexOfData[currentDataPointIndex] = connectedComponentNum;
		stack.push_back(currentDataPointIndex);
		while (stack.size() > 0)
		{
			int j = stack[stack.size() - 1];
			stack.pop_back();
			for (SparseMatrix<int>::InnerIterator it(L, j); it; ++it)
				if (!visit[it.row()] && it.value() == 4)
				{
					visit[it.row()] = true;
					stack.push_back(it.row());
					connectedComponentIndexOfData[it.row()] = connectedComponentNum;
				}
		}
		++connectedComponentNum;
	}
	cout << connectedComponentNum << endl;
	if (connectedComponentNum > 1)
	{
		vector<int> connectedComponentIndex(connectedComponentNum, 0);
		vector<vector<int>> componentPointIndex(connectedComponentNum);
		for (int i = 0; i < dataPointNum; ++i)
			componentPointIndex[connectedComponentIndexOfData[i]].push_back(i);
		/*long long distBetweenConnectedComponentsNum = (long long)connectedComponentNum* (connectedComponentNum - 1) / 2;
		vector<pair<double, pair<int, int>>> distBetweenConnectedComponents(distBetweenConnectedComponentsNum);
		cout << "L2" << endl;
#pragma omp parallel for schedule(dynamic, 32)
		for (int i = 0; i < connectedComponentNum; ++i)
		{
			if (i % 200 == 0)
				cout << i << endl;
			connectedComponentIndex[i] = i;
			for (int j = i + 1; j < connectedComponentNum; ++j)
			{
				pair<double, pair<int, int>> minDist(DBL_MAX, make_pair(-1, -1));
				for (int k = 0; k < componentPointIndex[i].size(); ++k)
				{
					for (int l = 0; l < componentPointIndex[j].size(); ++l)
					{
						double tempDist = distGraph.coeff(componentPointIndex[i][k], componentPointIndex[j][l]);
						if (tempDist == 0)
							tempDist = 1 - (double)getsnnNum(knnGraph, componentPointIndex[i][k], componentPointIndex[j][l]) / sharedNearestNeighborNum;
						if (tempDist < minDist.first)
						{
							minDist.first = tempDist;
							minDist.second = make_pair(componentPointIndex[i][k], componentPointIndex[j][l]);
						}
					}
				}
				long long id = (connectedComponentNum - 1) * (long long)i - (i + (long long)i * i) / 2 + j - 1;
				distBetweenConnectedComponents[id] = minDist;
			}
		}
		sort(distBetweenConnectedComponents.begin(), distBetweenConnectedComponents.end(), cmp);
		int interConnectedComponentEdgeNum = 0;
		int minDistIndex = 0, mergeLabel1, mergeLabel2;
		while (interConnectedComponentEdgeNum < connectedComponentNum - 1)
		{
			int tempRow, tempCol;
			tempRow = distBetweenConnectedComponents[minDistIndex].second.first;
			tempCol = distBetweenConnectedComponents[minDistIndex].second.second;
			while (connectedComponentIndex[connectedComponentIndexOfData[tempRow]]
				== connectedComponentIndex[connectedComponentIndexOfData[tempCol]])
			{
				++minDistIndex;
				tempRow = distBetweenConnectedComponents[minDistIndex].second.first;
				tempCol = distBetweenConnectedComponents[minDistIndex].second.second;
			}
			if (L.coeff(tempRow, tempCol) == 2 || L.coeff(tempCol, tempRow) == 2)
			{
				cout << "L(" << tempRow << ", " << tempCol << ") = " << L.coeff(tempRow, tempCol) << endl;
				exit(0);
			}
			L.coeffRef(tempRow, tempCol) = 2;
			L.coeffRef(tempCol, tempRow) = 2;
			distGraph.coeffRef(tempCol, tempRow) = 2 * distBetweenConnectedComponents[minDistIndex].first;
			distGraph.coeffRef(tempRow, tempCol) = 2 * distBetweenConnectedComponents[minDistIndex].first;
			mergeLabel1 = connectedComponentIndex[connectedComponentIndexOfData[tempRow]];
			mergeLabel2 = connectedComponentIndex[connectedComponentIndexOfData[tempCol]];
			for (int i = 0; i < connectedComponentNum; ++i)
				if (connectedComponentIndex[i] == mergeLabel1)
					connectedComponentIndex[i] = mergeLabel2;
			++interConnectedComponentEdgeNum;
		}*/
		/*for (int i = 0; i < connectedComponentNum; ++i)
		{
			connectedComponentIndex[i] = i;
		}
		int interConnectedComponentEdgeNum = 0;
		int minDistIndex = 0, mergeLabel1, mergeLabel2;
		while (interConnectedComponentEdgeNum < connectedComponentNum - 1)
		{

			int tempRow, tempCol;
			tempRow = distBetweenConnectedComponents[minDistIndex].second.first;
			tempCol = distBetweenConnectedComponents[minDistIndex].second.second;
			while (connectedComponentIndex[connectedComponentIndexOfData[tempRow]]
				== connectedComponentIndex[connectedComponentIndexOfData[tempCol]])
			{
				++minDistIndex;
				tempRow = distBetweenConnectedComponents[minDistIndex].second.first;
				tempCol = distBetweenConnectedComponents[minDistIndex].second.second;
			}
			if (L.coeff(tempRow, tempCol) == 2 || L.coeff(tempCol, tempRow) == 2)
			{
				cout << "L(" << tempRow << ", " << tempCol << ") = " << L.coeff(tempRow, tempCol) << endl;
				exit(0);
			}
			L.coeffRef(tempRow, tempCol) = 2;
			L.coeffRef(tempCol, tempRow) = 2;
			distGraph.coeffRef(tempCol, tempRow) = 2 * distBetweenConnectedComponents[minDistIndex].first;
			distGraph.coeffRef(tempRow, tempCol) = 2 * distBetweenConnectedComponents[minDistIndex].first;
			mergeLabel1 = connectedComponentIndex[connectedComponentIndexOfData[tempRow]];
			mergeLabel2 = connectedComponentIndex[connectedComponentIndexOfData[tempCol]];
			for (int i = 0; i < connectedComponentNum; ++i)
				if (connectedComponentIndex[i] == mergeLabel1)
					connectedComponentIndex[i] = mergeLabel2;
			++interConnectedComponentEdgeNum;
		}*/
	}
	cout << "L3" << endl;
	for (int i = 0; i < L.outerSize(); ++i)
	{
		int degree = 0;
		for (SparseMatrix<int>::InnerIterator it(L, i); it; ++it)
			if (it.value() > 1)
			{
				it.valueRef() = -1;
				++degree;
			}
			else
			{
				it.valueRef() = 0;
			}
		if (L.coeff(i, i) == -1)
			L.coeffRef(i, i) = degree - 1;
		else
			L.coeffRef(i, i) = degree;
	}
	L.prune(1, 0);
	return make_pair(L, distGraph);
}

vector<vector<double> > GetGreenFuncGrad(const SparseMatrix<int> &graphLaplacianMat, const SparseMatrix<double> &distGraph, int reservedEdgeNum)
{
	SparseMatrix<double> L = graphLaplacianMat.cast<double>();
	int dataPointNum = graphLaplacianMat.cols();
	/*vector<Triplet<double> > coefficients;
	vector<vector<Triplet<double> > > buffers;
	vector<vector<double> > gradMat;*/

	int threadNum = omp_get_max_threads();
	vector<vector<double> > gradient(dataPointNum);
	for (int i = 0; i < dataPointNum; ++i)
		gradient[i].resize(reservedEdgeNum);
	/*buffers.resize(threadNum);
	gradMat.resize(threadNum);
	for (int i = 0; i < threadNum; ++i)
		gradMat[i].resize(reservedEdgeNum);*/

	int edgeNum = (int)((graphLaplacianMat.nonZeros() - dataPointNum) / 2);
	vector<double> dist;
	for (int i = 0; i < graphLaplacianMat.outerSize(); ++i)
		for (SparseMatrix<int>::InnerIterator it(graphLaplacianMat, i); it; ++it)
			if (it.row() < it.col())
				dist.push_back(distGraph.coeff(it.row(), it.col()) / 2);
			else
				break;
	for (SparseMatrix<double>::InnerIterator it(L, 0); it; ++it)
		L.coeffRef(it.row(), it.col()) = L.coeffRef(it.col(), it.row()) = 0;
	L.coeffRef(0, 0) = 1;
	SimplicialLDLT<SparseMatrix<double> > solver;
	solver.compute(L);
	srand(0);
	int sampleNum = min(dataPointNum / 10, 1000);
	VectorXd sum(edgeNum);
	sum.setZero();
	vector<pair<int, int>> edgeID(edgeNum);
	vector<int> samples;
	while (samples.size() < sampleNum)
	{
		int id = ((double)rand() / RAND_MAX) * dataPointNum;
		if (find(samples.begin(), samples.end(), id) == samples.end())
			samples.push_back(id);
		else
			continue;

		VectorXd b(dataPointNum);
		VectorXd sub(dataPointNum);
		b.fill(-(double)1 / dataPointNum);
		b[id] += 1;
		b[0] = 0;
		VectorXd gi = solver.solve(b);
		sub.fill(gi.mean());
		gi -= sub;
		int count = 0;

		VectorXd tempGrad(edgeNum);
		for (int j = 0; j < graphLaplacianMat.outerSize(); ++j)
		{
			for (SparseMatrix<int>::InnerIterator it(graphLaplacianMat, j); it; ++it)
			{
				if (it.row() < it.col())
				{
					tempGrad(count) = abs((gi[it.col()] - gi[it.row()]) * dist[count]);
					edgeID[count] = make_pair(it.col(), it.row());
					++count;
				}
				else
					break;
			}
		}
		sum += tempGrad;
	}
	size_t reservedEle = static_cast<size_t>(reservedEdgeNum);
	auto comp = [](const pair<double, int>& e1, const pair<double, int>& e2) {
		return e1.first > e2.first;
	};
	priority_queue<pair<double, int>, vector<pair<double, int>>, decltype(comp)> heap(comp);
	for (int i = 0; i < sum.rows(); ++i)
	{
		if (heap.size() >= reservedEle)
		{
			if (heap.top().first < sum[i])
			{
				heap.pop();
				heap.push(make_pair(sum[i], i));
			}
		}
		else
			heap.push(make_pair(sum[i], i));
	}
	vector<int> indices;
	while (!heap.empty())
	{
		indices.push_back(heap.top().second);
		heap.pop();
	}
	sort(indices.begin(), indices.end());
	
#pragma omp parallel for
	for (int i = 0; i < dataPointNum; ++i)
	{
		if (i % 200 == 0)
			cout << i << endl;
		int count = 0;
		auto id = omp_get_thread_num();
		VectorXd b(dataPointNum);
		VectorXd sub(dataPointNum);
		VectorXd gi(dataPointNum);
		b.fill(-(double)1 / dataPointNum);
		b[i] += 1;
		b[0] = 0;
		//g.col(i) = solver.solve(b);
		gi = solver.solve(b);
		sub.fill(gi.mean());
		gi -= sub;

		/*for (size_t j = 0; j < indices.size(); ++j)
			gradMat[id][j] = (gi[edgeID[indices[j]].first] - gi[edgeID[indices[j]].second])*dist[indices[j]];

		for (int j = 0; j < reservedEdgeNum; ++j)
			buffers[id].push_back(Triplet<double>(i, j, gradMat[id][j]));*/

		for (int j = 0; j < indices.size(); ++j)
			gradient[i][j] = (gi[edgeID[indices[j]].first] - gi[edgeID[indices[j]].second])*dist[indices[j]];
	}
	/*for (auto & buffer : buffers)
		move(buffer.begin(), buffer.end(), back_inserter(coefficients));*/

	/*SparseMatrix<double, RowMajor> gradient(dataPointNum, indices.size());
	gradient.setFromTriplets(coefficients.begin(), coefficients.end());
	gradient.makeCompressed();*/
	return gradient;
}

vector<vector<double> > kMeansPlusPlusInit(const vector<vector<double> > &data, int clusterNum)
{
	vector<vector<double> > centroids(clusterNum);
	if (clusterNum < 1)
		return centroids;
	int rowNum = data.size(), colNum = data[0].size(), centNum;
	vector<double> dist(rowNum);
	centroids[0] = data[(int)(((double)rand() / RAND_MAX) * rowNum)];
	centNum = 1;
	while (centNum < clusterNum)
	{
		fill(dist.begin(), dist.end(), DBL_MAX);
		for(int i = 0;i<rowNum; ++i)
			for (int j = 0; j < centNum; ++j)
			{
				double tempDist = 0;
				for (int k = 0; k < colNum; ++k)
					tempDist += pow(centroids[j][k] - data[i][k], 2);
				if (tempDist < dist[i])
					dist[i] = tempDist;
			}
		double totalDist = accumulate(dist.begin(), dist.end(), 0.0);
		for (int i = 0; i < rowNum; ++i)
			dist[i] /= totalDist;
		for (int i = 1; i < rowNum; ++i)
			dist[i] += dist[i - 1];
		double randProb = (double)rand() / RAND_MAX;
		for (int i = 0; i < rowNum; ++i)
			if (randProb <= dist[i])
			{
				centroids[centNum] = data[i];
				break;
			}
		++centNum;
	}
	return centroids;
}

vector<int> kMeans(const vector<vector<double> > &data, int clusterNum, int initNum)
{
	int rowNum = data.size(), colNum = data[0].size(), threadNum = omp_get_max_threads();
	vector<int> labels(rowNum);
	int iterNum = 0;
	double inertia = DBL_MAX;
	while (iterNum < initNum)
	{
		vector<vector<double> > centroids;
		centroids = kMeansPlusPlusInit(data, clusterNum);
		bool clusterChange;
		vector<bool> tempChang(threadNum, false);
		clusterChange = true;
		vector<int> tempLabels(rowNum, 0);
		double tempInertia = 0;
		vector<double> dataPointInertia(threadNum, 0);
		while (clusterChange)
		{
			fill(tempChang.begin(), tempChang.end(), false);
			clusterChange = false;
#pragma omp parallel for
			for (int i = 0; i < rowNum; ++i)
			{
				auto id = omp_get_thread_num();
				double minDist = DBL_MAX;
				int minIndex = -1;
				for (int j = 0; j < clusterNum; ++j)
				{
					double tempDist = 0;
					for (int k = 0; k < colNum; ++k)
						tempDist += pow(data[i][k] - centroids[j][k], 2);
					if (tempDist < minDist)
					{
						minDist = tempDist;
						minIndex = j;
					}
				}
					dataPointInertia[id] += minDist;
				if (tempLabels[i] != minIndex)
				{
					tempChang[id] = true;
					tempLabels[i] = minIndex;
				}
			}
			for (int i = 0; i < clusterNum; ++i)
			{
				int count = 0;
				vector<double> totalData(colNum, 0);
				for (int j = 0; j < rowNum; ++j)
					if (tempLabels[j] == i)
					{
						++count;
						for (int k = 0; k < colNum; ++k)
							totalData[k] += data[j][k];
					}
				if (count == 0)
					centroids[i] = (kMeansPlusPlusInit(data, 1))[0];
				else
					for (int j = 0; j < colNum; ++j)
						centroids[i][j] = totalData[j] / count;
			}
			for (int i = 0; i < threadNum; ++i)
				if (tempChang[i])
				{
					clusterChange = true;
					break;
				}
		}
		tempInertia = accumulate(dataPointInertia.begin(), dataPointInertia.end(), 0.0);
		if (tempInertia < inertia)
		{
			inertia = tempInertia;
			labels = tempLabels;
		}
		++iterNum;
	}
	return labels;
}