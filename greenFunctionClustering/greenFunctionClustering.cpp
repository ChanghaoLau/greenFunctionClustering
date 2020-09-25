#include <omp.h>
#include <queue>
#include <vector>
#include <time.h>
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

struct arrWithIndex
{
	double data;
	int index;
};

struct L2WithIndex
{
	double dist;
	int row;
	int col;
};

vector<double *> dataMat;
//double compressedRatio;
//double simcos_threshold;
int n_data, n_dimension, kNeighborNumber, n_cluster, snn;


void loadDataSet(char *fileName);
//bool cmp(const arrWithIndex a, const arrWithIndex b);
//bool absCmp(const arrWithIndex a, const arrWithIndex b);
//void argsort(const vector<double> &arr, vector<int> &sortedIndex, bool absolutFlag);
void constructHeap(arrWithIndex a[], int n, arrWithIndex value);
void UpdataHeap(arrWithIndex a[], int index, int n);
void topKIndex(double arr[], int n, int sortedIndex[], int k, bool absoluteFlag, bool largestFlag);
double disKNN(const SparseMatrix<double>& KNN, int i, int j);
SparseMatrix<double> calcLaplacianMat();
SparseMatrix<double> calcSNNLaplacianMat();
SparseMatrix<double> calcLaplaceMat();
bool cmp(const L2WithIndex a, const L2WithIndex b);
SparseMatrix<double> constructLaplaceMat();
int connectedComponent(const SparseMatrix<double> &Laplace);
SparseMatrix<double, RowMajor> calcGreenFunctionGradient(const SparseMatrix<double> &Laplace);
//SparseMatrix<double, RowMajor> getGradient(const SparseMatrix<double> &Laplace, const vector<VectorXd> &g, double ratio);
void kMeansPlusPlusInitWithCSRMat(double **centroids, const SparseMatrix<double, RowMajor> &data, int n_clusters);
void kMeansWithCSRMat(int *clusterAssessment, const SparseMatrix<double, RowMajor> &data, int n_clusters);
void storeOutcome(char *fileName, int *assesment, clock_t t1, clock_t t2, clock_t t3, clock_t t4);

int main()
{
	int i, j;
	double degree;
	vector<VectorXd> g;
	clock_t t1, t2, t3, t4;
	SparseMatrix<double> L, L0;
	SparseMatrix<double, RowMajor> grad;
	char fileName[51];
	int *assessment;
	FILE *fp;
	errno_t err;
	printf("file name(no more than 50 characters):");
	scanf_s("%s", fileName, 51);
	printf("the number of data points:");
	scanf_s("%d", &n_data);
	printf("data dimension:");
	scanf_s("%d", &n_dimension);
	printf("knn:");
	scanf_s("%d", &kNeighborNumber);
	printf("SNN:");
	scanf_s("%d", &snn);
	/*printf("simcos threshold:");
	scanf_s("%lf", &simcos_threshold);*/
	/*printf("compression ratio:");
	scanf_s("%lf", &compressedRatio);*/
	printf("the number of clusters:");
	scanf_s("%d", &n_cluster);
	assessment = (int *)calloc(n_data, sizeof(int));
	loadDataSet(fileName);
	t1 = clock();
	L = calcSNNLaplacianMat();
	/*if (err = fopen_s(&fp, "L.txt", "w"))
	{
		printf("timeFile error value: %d", err);
		exit(1);
	}
	for (i = 0; i < L.rows(); ++i)
	{
		for (j = 0; j < L.cols(); ++j)
			fprintf(fp, "%.18f\t", L.coeff(i, j));
		fprintf(fp, "\n");
	}
	fclose(fp);*/
	t2 = clock();

	/*int *labels = (int *)calloc(n_data, sizeof(int));
	if (err = fopen_s(&fp, "complexStructureLabels.txt", "r"))
	{
		printf("timeFile error value: %d", err);
		exit(1);
	}
	for (i = 0; i < n_data; ++i)
		fscanf_s(fp, "%d", &labels[i]);
	fclose(fp);
	
	int intraClassEdges = 0;
	int interClassEdges = 0;
	double intraClassEdgesLen = 0;
	double interClassEdgesLen = 0;*/
	///*for (i = 0; i < L.outerSize(); ++i)
	//{
	//	for (SparseMatrix<double>::InnerIterator it(L, i); it; ++it)
	//	{
	//		if (it.value() == -1)
	//		{
	//			++intraClassEdges;
	//		}
	//	}
	//}
	//printf("%d\n", intraClassEdges);
	//intraClassEdges = 0;
	//for (i = 0; i < L.outerSize(); ++i)
	//	if(L.coeff(i, i) == 0)
	//		++intraClassEdges;
	//printf("%d\n", intraClassEdges);
	//intraClassEdges = 0;
	//for (i = 0; i < L.outerSize(); ++i)
	//{
	//	for (SparseMatrix<double>::InnerIterator it(L, i); it; ++it)
	//	{
	//		if (it.value() == -1)
	//		{
	//			if(it.row() < i)
	//				++intraClassEdges;
	//			if (it.row() > i)
	//				++interClassEdges;
	//		}
	//	}
	//}
	//printf("%d\t%d\n", intraClassEdges, interClassEdges);
	//intraClassEdges = 0;
	//interClassEdges = 0;*/
	//for (i = 0; i < L.outerSize(); ++i)
	//{
	//	int curLabel = labels[i];
	//	for (SparseMatrix<double>::InnerIterator it(L, i); it.row() < i; ++it)
	//	{
	//		if (it.value() == -1)
	//		{
	//			double distIJ = 0;
	//			for (j = 0; j < n_dimension; ++j)
	//				distIJ += pow(dataMat[i][j] - dataMat[it.row()][j], 2);
	//			distIJ = pow(distIJ, 0.5);
	//			if (labels[it.row()] == curLabel)
	//			{
	//				++intraClassEdges;
	//				intraClassEdgesLen += distIJ;
	//			}
	//			else
	//			{
	//				//printf("%d\t%d\n", i, it.row());
	//				++interClassEdges;
	//				interClassEdgesLen += distIJ;
	//			}
	//		}
	//	}
	//}
	//printf("%d\t%d\t%d\t%d\n", intraClassEdges, interClassEdges, (int)((L.nonZeros() - n_data) / 2), L.nonZeros());
	//printf("%f\t%f\n", intraClassEdgesLen / intraClassEdges, interClassEdgesLen / interClassEdges);

	printf("L:%f\n", (double)(t2 - t1) / CLOCKS_PER_SEC);
	//printf("%d\n", connectedComponent(L));
	if (connectedComponent(L) > 1)
	{
		printf("Laplacian Matrix is not a connected graph.\n");
		exit(0);
	}
	grad = calcGreenFunctionGradient(L);
	
	/*for (i = 0; i < grad.outerSize(); ++i)
		for (SparseMatrix<double, RowMajor>::InnerIterator it(grad, i); it; ++it)
			fprintf(fp, "%f\t%d\t%d\n", it.value(), i, it.col());*/

	/*for (i = 0; i < 10; ++i)
		for (SparseMatrix<double, RowMajor>::InnerIterator it(grad, i); it; ++it)
			printf("%f\t%d\t%d\n", it.value(), i, it.col());*/
	
	/*grad = getGradient(L, g, compressedRatio);*/
	t3 = clock();
	printf("grad:%f\n", (double)(t3 - t2) / CLOCKS_PER_SEC);
	/*kMeansWithCSRMat(assessment, grad, n_cluster);
	t4 = clock();
	printf("kmeans:%f\n", (double)(t4 - t3) / CLOCKS_PER_SEC);
	printf("total:%f\n", (double)(t4 - t1) / CLOCKS_PER_SEC);
	storeOutcome(fileName, assessment, t1, t2, t3, t4);*/

	
	if (err = fopen_s(&fp, "grad.txt", "w"))
	{
		printf("timeFile error value: %d", err);
		exit(1);
	}
	for (i = 0; i < grad.rows(); ++i)
	{
		for (j = 0;j< grad.cols();++j)
			fprintf(fp, "%.18f\t", grad.coeff(i, j));
		fprintf(fp, "\n");
	}
	fclose(fp);
	//char *cmd = (char *)malloc(50 * sizeof(char));
	//sprintf_s(cmd, 50 * sizeof(char), "%s", "python reuters10k_acc.py");
	////printf("%s\n", cmd);
	//system(cmd);
	//free(cmd);

	for (i = 0; i < n_data; ++i)
		free(dataMat[i]);
	free(assessment);

	//int i, j;
	//char fileName[51];
	//FILE *fp;
	//errno_t err;
	//printf("file name(no more than 50 characters):");
	//scanf_s("%s", fileName, 51);
	//printf("the number of data points:");
	//scanf_s("%d", &n_data);
	//printf("data dimension:");
	//scanf_s("%d", &n_dimension);
	//printf("the number of clusters:");
	//scanf_s("%d", &n_cluster);
	//loadDataSet(fileName);
	//for (snn = 400;snn<=1500;snn+=50)
	//for (kNeighborNumber = 2; kNeighborNumber <= 12; kNeighborNumber+=1)
	//{
	//	printf("%d\t%d:\n", snn, kNeighborNumber);
	//	SparseMatrix<double> L;
	//	SparseMatrix<double, RowMajor> grad;
	//	L = calcSNNLaplacianMat();
	//	if (connectedComponent(L) > 1)
	//	{
	//		printf("Laplacian Matrix is not a connected graph: %d\n", kNeighborNumber);
	//		exit(0);
	//	}
	//	grad = calcGreenFunctionGradient(L);
	//	if (err = fopen_s(&fp, "grad.txt", "w"))
	//	{
	//		printf("timeFile error value: %d", err);
	//		exit(1);
	//	}
	//	for (i = 0; i < grad.rows(); ++i)
	//	{
	//		for (j = 0; j < grad.cols(); ++j)
	//			fprintf(fp, "%.18f\t", grad.coeff(i, j));
	//		fprintf(fp, "\n");
	//	}
	//	fclose(fp);
	//	//system("python reuters10k_acc.py");
	//	char *cmd = (char *)malloc(50 * sizeof(char));
	//	sprintf_s(cmd, 50 * sizeof(char), "%s %d %d", "python reuters10k_acc.py", snn, kNeighborNumber);
	//	//printf("%s\n", cmd);
	//	system(cmd);
	//	free(cmd);
	//}
	//for (i = 0; i < n_data; ++i)
	//	free(dataMat[i]);

	getchar();
	return 0;
}

void loadDataSet(char *fileName)
{
	int i, j;
	FILE *fp;
	errno_t err;
	//printf("Please enter your data file name(no more than 50 characters):");
	dataMat.resize(n_data);
	for (i = 0; i < n_data; ++i)
		dataMat[i] = (double *)malloc(n_dimension * sizeof(double));
	if (err = fopen_s(&fp, fileName, "r"))
	{
		printf("error value: %d", err);
		exit(1);
	}
	for (i = 0; i < n_data; ++i)
		for (j = 0; j < n_dimension; ++j)
			fscanf_s(fp, "%lf", &dataMat[i][j]);
	fclose(fp);
}

//bool cmp(const arrWithIndex a, const arrWithIndex b)
//{
//	if (a.data != b.data)
//		return a.data < b.data;
//	else
//		return a.index < b.index;
//}
//
//bool absCmp(const arrWithIndex a, const arrWithIndex b)
//{
//	if (abs(a.data) != abs(b.data))
//		return abs(a.data) < abs(b.data);
//	else
//		return a.index < b.index;
//}
//
//void argsort(const vector<double> &arr, vector<int> &sortedIndex, bool absolutFlag = 0)
//{
//	int i, n = arr.size();
//	/*arrWithIndex *a = (arrWithIndex *)malloc(n * sizeof(arrWithIndex));
//	for (i = 0; i < n; ++i)
//	{
//		a[i].data = arr[i];
//		a[i].index = i;
//	}
//	if (absolutFlag)
//		sort(a, a + n, absCmp);
//	else
//		sort(a, a + n, cmp);*/
//	vector<arrWithIndex> a(n);
//	for (i = 0; i < n; ++i)
//	{
//		a[i].data = arr[i];
//		a[i].index = i;
//	}
//	if (absolutFlag)
//		sort(a.begin(), a.end(), absCmp);
//	else
//		sort(a.begin(), a.end(), cmp);
//	for (i = 0; i < n; ++i)
//		sortedIndex[i] = a[i].index;
//}

void constructHeap(arrWithIndex a[], int n, arrWithIndex value) 
{
	a[n] = value;
	int j;
	arrWithIndex temp = value;
	j = (n - 1) / 2;
	while (j >= 0 && n != 0)
	{
		if (a[j].data < temp.data)
			break;
		a[n] = a[j];
		n = j;
		j = (n - 1) / 2;
	}
	a[n] = temp;
}

void UpdataHeap(arrWithIndex a[], int index, int n)
{
	int j;
	arrWithIndex temp = a[index];
	j = 2 * index + 1;
	while (j < n)
	{
		if (j + 1 < n && a[j + 1].data <= a[j].data)
			++j;
		if (a[j].data >= temp.data)
			break;
		a[index] = a[j];
		index = j;
		j = index * 2 + 1;
	}
	a[index] = temp;
}

void topKIndex(double arr[], int n, int sortedIndex[], int k, bool absoluteFlag, bool largestFlag) {

	int i;
	arrWithIndex *a = (arrWithIndex *)malloc(n * sizeof(arrWithIndex));
	arrWithIndex *temp = (arrWithIndex *)malloc(k * sizeof(arrWithIndex));
	for (i = 0; i < n; ++i)
	{
		a[i].data = arr[i];
		a[i].index = i;
	}
	if (absoluteFlag)
		for (i = 0; i < n; ++i)
			a[i].data = abs(a[i].data);
	if (!largestFlag)
		for (i = 0; i < n; ++i)
			a[i].data = -a[i].data;
	for (i = 0; i < n; ++i)
	{
		if (i < k)
			constructHeap(temp, i, a[i]);
		else
		{
			if (temp[0].data < a[i].data)
			{
				temp[0] = a[i];
				UpdataHeap(temp, 0, k);
			}
		}
	}
	for (i = 0; i < k; ++i)
		sortedIndex[i] = temp[i].index;
	free(temp);
	free(a);
}

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
int KDTreePoint::DIM = 1;

double disKNN(const SparseMatrix<double>& KNN, int i, int j)
{
	int n_knn = 0;
	for (SparseMatrix<double>::InnerIterator it(KNN, i); it; ++it)
		if (KNN.coeff(it.row(), j) == 1)
			++n_knn;
	return n_knn;
}

SparseMatrix<double> calcLaplacianMat()
{
	int i, j, degree, n_connectedComponent, top, interConnectedComponentEdge;
	vector<Triplet<double>> coefficients;
	vector<vector<Triplet<double> > > buffers;
	SparseMatrix<double> L(n_data, n_data);

	int n_threads = omp_get_max_threads();
	buffers.resize(n_threads);
	//distI = vector<vector<double> >(n_threads, vector<double>(n_data));
	//distWithIndex = vector<vector<int> >(n_threads, vector<int>(n_data));
	//printf("Start parallel: %d threads\n", n_threads);

	vector<KDTreePoint> points(dataMat.begin(), dataMat.end());
	KDTree<KDTreePoint> kdtree(points);
	KDTreePoint::DIM = n_dimension;

	/*FILE *fp;
	errno_t err;
	int *labels = (int *)calloc(n_data, sizeof(int));
	if (err = fopen_s(&fp, "complexStructureLabels.txt", "r"))
	{
		printf("timeFile error value: %d", err);
		exit(1);
	}
	for (i = 0; i < n_data; ++i)
		fscanf_s(fp, "%d", &labels[i]);
	fclose(fp);*/

#pragma omp parallel for
	for (int i = 0; i < n_data; ++i)
	{
		/*if (i % 1000 == 0)
			printf("L1: %d\n", i);*/

		KDTreePoint p(dataMat[i]);
		vector<int> knn_ind = kdtree.knnSearch(p, kNeighborNumber + 1);
		/*if (knn_ind[0] != i)
			printf("%\n", i);*/
		auto id = omp_get_thread_num();
		for (size_t j = 0; j < knn_ind.size(); ++j)
		{
			/*if (labels[i] != labels[knn_ind[j]] && rand() < RAND_MAX * 0.95)
				continue;*/
			buffers[id].push_back(Triplet<double>(i, knn_ind[j], 1));
			buffers[id].push_back(Triplet<double>(knn_ind[j], i, 1));
		}
	}

	for (auto & buffer : buffers)
	{
		move(buffer.begin(), buffer.end(), back_inserter(coefficients));
	}

//	double **distI;
//	int **distWithIndex;
//	int n_threads;
//
//#pragma omp parallel
//	{
//#pragma omp single
//		{
//			n_threads = omp_get_max_threads();
//			distI = (double **)malloc(n_threads * sizeof(double *));
//			for (i = 0; i < n_threads; ++i)
//				distI[i] = (double *)malloc(n_data * sizeof(double));
//			distWithIndex = (int **)malloc(n_threads * sizeof(int *));
//			for (i = 0; i < n_threads; ++i)
//				distWithIndex[i] = (int *)malloc((kNeighborNumber + 1) * sizeof(int));
//			/*vector<vector<double> > distI(n_threads, vector<double>(n_data));
//			vector<vector<int> > distWithIndex(n_threads, vector<int>(n_data));*/
//			//#pragma omp master
//			buffers.resize(n_threads);
//			//distI = vector<vector<double> >(n_threads, vector<double>(n_data));
//			//distWithIndex = vector<vector<int> >(n_threads, vector<int>(n_data));
//			printf("Start parallel: %d threads\n", n_threads);
//		}
//#pragma omp for private(j) private(k)
//		for (i = 0; i < n_data; ++i)
//		{
//			if (i % 1000 == 0)
//				printf("L1: %d\n", i);
//			auto id = omp_get_thread_num();
//			for (j = 0; j < n_data; ++j)
//			{
//				double distIJ = 0;
//				for (k = 0; k < n_dimension; ++k)
//					distIJ += pow(dataMat[i][k] - dataMat[j][k], 2);
//				distIJ = pow(distIJ, 0.5);
//				distI[id][j] = distIJ;
//			}
//			topKIndex(distI[id], n_data, distWithIndex[id], kNeighborNumber + 1, 0, 0);
//			for (j = 0; j <= kNeighborNumber; ++j)
//			{
//				buffers[id].push_back(Triplet<double>(i, distWithIndex[id][j], 1));
//				buffers[id].push_back(Triplet<double>(distWithIndex[id][j], i, 1));
//			}
//		}
//
//#pragma omp master
//		{
//			for (auto & buffer : buffers)
//			{
//				move(buffer.begin(), buffer.end(), back_inserter(coefficients));
//			}
//			for (i = 0; i < n_threads; ++i)
//				free(distI[i]);
//			free(distI);
//			for (i = 0; i < n_threads; ++i)
//				free(distWithIndex[i]);
//			free(distWithIndex);
//			printf("End parallel\n");
//		}
//	}
	L.setFromTriplets(coefficients.begin(), coefficients.end());
	for (i = 0; i < L.outerSize(); ++i)
	{
		degree = 0;
		for (SparseMatrix<double>::InnerIterator it(L, i); it; ++it)
			if (it.value() > 0)
			{
				//printf("%d\t%d\t%f\t", it.col(), it.row(), it.value());
				it.valueRef() = -1;
				++degree;
			}
			else
			{
				//printf("%d\t%d\t%f\t", i, it.row(), it.value());
				it.valueRef() = 0;
			}
		if(L.coeffRef(i, i)==-1)
			L.coeffRef(i, i) = degree - 1;
		else
			L.coeffRef(i, i) = degree;
	}
	L.prune(1, 1e-8);

	/*FILE *fp;
	errno_t err;
	if (err = fopen_s(&fp, "L.txt", "w"))
	{
		printf("timeFile error value: %d", err);
		exit(1);
	}
	for (i = 0; i < L.rows(); ++i)
	{
		for (j = 0; j < L.cols(); ++j)
			fprintf(fp, "%.18f\t", L.coeff(i, j));
		fprintf(fp, "\n");
	}
	fclose(fp);*/

	i = 0;
	n_connectedComponent = 0;
	top = -1;
	int *dataConnectedComponentIndex = (int *)calloc(n_data, sizeof(int));
	int *visit = (int *)calloc(n_data, sizeof(int));
	int *stack = (int *)calloc(n_data, sizeof(int));
	while (1)
	{
		while (i < n_data && visit[i])
			++i;
		if (i == n_data)
			break;
		visit[i] = 1;
		stack[++top] = i;
		dataConnectedComponentIndex[i] = n_connectedComponent;
		while (top > -1)
		{
			j = stack[top--];
			for (SparseMatrix<double>::InnerIterator it(L, j); it; ++it)
			{
				if (it.value() == -1 && visit[it.row()] == 0)
				{
					visit[it.row()] = 1;
					stack[++top] = it.row();
					dataConnectedComponentIndex[it.row()] = n_connectedComponent;
				}
			}
		}
		++n_connectedComponent;
	}

	/*int count0 = 0;
	FILE *fp;
	errno_t err;
	if (err = fopen_s(&fp, "Component.txt", "w"))
	{
		printf("timeFile error value: %d", err);
		exit(1);
	}
	for (i = 0; i < n_connectedComponent; ++i)
	{
		count0 = 0;
		for (j = 0; j < n_data; ++j)
			if (dataConnectedComponentIndex[j] == i)
				++count0;
		fprintf(fp, "%d: %d\n", i, count0);
	}
	fclose(fp);*/

	free(stack);
	free(visit);
	printf("the number of connected component of mutual knn graph: %d\n", n_connectedComponent);
	if (n_connectedComponent == 1)
		return L;
	int *connectedComponentIndex = (int *)calloc(n_connectedComponent, sizeof(int));
	L2WithIndex *distP = (L2WithIndex *)malloc(n_connectedComponent * (n_connectedComponent - 1) / 2 * sizeof(L2WithIndex));

	//construct kdtree
	vector<vector<KDTreePoint>> component_point(n_connectedComponent);
	vector<vector<int>> component_point_id(n_connectedComponent);
	for (int i = 0; i < n_data; ++i)
	{
		int id = dataConnectedComponentIndex[i];
		component_point[id].push_back(KDTreePoint(dataMat[i]));
		component_point_id[id].push_back(i);
	}
	vector<KDTree<KDTreePoint>> component_kdtree(component_point.begin(), component_point.end());

#pragma omp parallel for schedule(dynamic, 8) private(j)
	for (i = 0; i < n_connectedComponent; ++i)
	{
		/*if (i % 100 == 0)
			printf("L2: %d\n", i);*/
		connectedComponentIndex[i] = i;
		for (j = i + 1; j < n_connectedComponent; ++j)
		{
			L2WithIndex minDist;
			minDist.dist = DBL_MAX;
			minDist.row = -1;
			minDist.col = -1;

			for (size_t k = 0; k < component_point[i].size(); ++k)
			{
				int id = component_kdtree[j].nnSearch(component_point[i][k]);
				double tempDist = 0;
				for (int l = 0; l < n_dimension; ++l)
					tempDist += pow(component_point[i][k][l] - component_point[j][id][l], 2);
				if (tempDist < minDist.dist)
				{
					minDist.dist = tempDist;
					minDist.row = component_point_id[i][k];
					minDist.col = component_point_id[j][id];
				}
			}
			minDist.dist = sqrt(minDist.dist);

			int index = (n_connectedComponent - 1) * i - (i + i * i) / 2 + j - 1;
			distP[index] = minDist;
		}
	}

//#pragma omp parallel for schedule(dynamic, 8) private(j) private(k)
//	for (i = 0; i < n_connectedComponent; ++i)
//	{
//		if (i % 100 == 0)
//			printf("L2: %d\n", i);
//		connectedComponentIndex[i] = i;
//		for (j = i + 1; j < n_connectedComponent; ++j)
//		{
//			L2WithIndex minDist;
//			minDist.dist = DBL_MAX;
//			minDist.row = -1;
//			minDist.col = -1;
//			for (int data_i = 0; data_i < n_data; ++data_i)
//				if (dataConnectedComponentIndex[data_i] == i)
//					for (int data_j = 0; data_j < n_data; ++data_j)
//					{
//						if (dataConnectedComponentIndex[data_j] == j)
//						{
//							double tempDist = 0;
//							for (k = 0; k < n_dimension; ++k)
//								tempDist += pow(dataMat[data_i][k] - dataMat[data_j][k], 2);
//							tempDist = pow(tempDist, 0.5);
//							if (tempDist < minDist.dist)
//							{
//								minDist.dist = tempDist;
//								minDist.row = data_i;
//								minDist.col = data_j;
//							}
//
//						}
//					}
//			int index = (n_connectedComponent - 1) * i - (i + i * i) / 2 + j - 1;
//			distP[index] = minDist;
//		}
//	}
	/*printf("%d\t%d\t%f\n", distP[0].row, distP[0].col, distP[0].dist);
	printf("%d\t%d\t%f\n", distP[1].row, distP[1].col, distP[1].dist);
	printf("%d\t%d\t%f\n", distP[2].row, distP[2].col, distP[2].dist);
	printf("%d\t%d\t%f\n", distP[50].row, distP[50].col, distP[50].dist);
	printf("%d\t%d\t%f\n", distP[100].row, distP[100].col, distP[100].dist);*/
	sort(distP, distP + (n_connectedComponent * (n_connectedComponent - 1) / 2), cmp);
	//printf("sort finish\n");
	interConnectedComponentEdge = 0;
	int minDistIndex = 0, mergeLabel1, mergeLabel2;
	while (interConnectedComponentEdge < n_connectedComponent - 1)
	{
		while (connectedComponentIndex[dataConnectedComponentIndex[distP[minDistIndex].row]] ==
			connectedComponentIndex[dataConnectedComponentIndex[distP[minDistIndex].col]])
			++minDistIndex;
		if (L.coeffRef(distP[minDistIndex].row, distP[minDistIndex].col) != 0 ||
			L.coeffRef(distP[minDistIndex].col, distP[minDistIndex].row) != 0)
		{
			printf("L(%d, %d) = %f", distP[minDistIndex].row, distP[minDistIndex].col,
				L.coeffRef(distP[minDistIndex].row, distP[minDistIndex].col));
			exit(0);
		}
		L.coeffRef(distP[minDistIndex].row, distP[minDistIndex].col) = -1;
		L.coeffRef(distP[minDistIndex].col, distP[minDistIndex].row) = -1;
		L.coeffRef(distP[minDistIndex].row, distP[minDistIndex].row) += 1;
		L.coeffRef(distP[minDistIndex].col, distP[minDistIndex].col) += 1;
		mergeLabel1 = connectedComponentIndex[dataConnectedComponentIndex[distP[minDistIndex].row]];
		mergeLabel2 = connectedComponentIndex[dataConnectedComponentIndex[distP[minDistIndex].col]];
		for (i = 0; i < n_connectedComponent; ++i)
			if (connectedComponentIndex[i] == mergeLabel1)
				connectedComponentIndex[i] = mergeLabel2;
		++interConnectedComponentEdge;
	}
	free(dataConnectedComponentIndex);
	free(connectedComponentIndex);
	free(distP);
	L.prune(1, 1e-8);
	return L;
}

SparseMatrix<double> calcSNNLaplacianMat()
{
	int i, j, degree, n_connectedComponent, top, interConnectedComponentEdge;
	vector<Triplet<double>> coefficients;
	vector<vector<Triplet<double> > > buffers;
	SparseMatrix<double> L(n_data, n_data);
	SparseMatrix<double> KNN(n_data, n_data);

	int n_threads = omp_get_max_threads();
	buffers.resize(n_threads);
	//printf("Start parallel: %d threads\n", n_threads);

	vector<KDTreePoint> points(dataMat.begin(), dataMat.end());
	KDTree<KDTreePoint> kdtree(points);
	KDTreePoint::DIM = n_dimension;

#pragma omp parallel for
	for (int i = 0; i < n_data; ++i)
	{
		/*if (i % 1000 == 0)
			printf("L1: %d\n", i);*/

		KDTreePoint p(dataMat[i]);
		vector<int> knn_ind = kdtree.knnSearch(p, snn + 1);
		auto id = omp_get_thread_num();
		for (size_t j = 0; j < knn_ind.size(); ++j)
		{
			/*if (labels[i] != labels[knn_ind[j]] && rand() < RAND_MAX * 0.95)
				continue;*/
			//buffers[id].push_back(Triplet<double>(i, knn_ind[j], 1));
			if (knn_ind[j] != i)
				buffers[id].push_back(Triplet<double>(knn_ind[j], i, 1));
		}
	}

	for (auto & buffer : buffers)
	{
		move(buffer.begin(), buffer.end(), back_inserter(coefficients));
	}
	KNN.setFromTriplets(coefficients.begin(), coefficients.end());

	/*FILE *fp;
	errno_t err;
	if (err = fopen_s(&fp, "KNN.txt", "w"))
	{
		printf("timeFile error value: %d", err);
		exit(1);
	}
	for (i = 0; i < KNN.rows(); ++i)
	{
		for (j = 0; j < KNN.cols(); ++j)
			fprintf(fp, "%.18f\t", KNN.coeff(i, j));
		fprintf(fp, "\n");
	}
	fclose(fp);*/

	//printf("%d\n", KNN.nonZeros());
	coefficients.clear();
	for (i = 0; i < n_threads; ++i)
		buffers[i].clear();

	double **distI;
	int **distWithIndex;

#pragma omp parallel
	{
#pragma omp single
		{
			distI = (double **)malloc(n_threads * sizeof(double *));
			for (i = 0; i < n_threads; ++i)
				distI[i] = (double *)malloc(n_data * sizeof(double));
			distWithIndex = (int **)malloc(n_threads * sizeof(int *));
			for (i = 0; i < n_threads; ++i)
				distWithIndex[i] = (int *)malloc((kNeighborNumber + 1) * sizeof(int));
			buffers.resize(n_threads);
			//printf("Start parallel: %d threads\n", n_threads);
		}
#pragma omp for private(j)
		for (i = 0; i < n_data; ++i)
		{
			/*if (i % 1000 == 0)
				printf("L1: %d\n", i);*/
			auto id = omp_get_thread_num();
			for (j = 0; j < n_data; ++j)
			{
				int n_knn = 0;
				for (SparseMatrix<double>::InnerIterator it(KNN, i); it; ++it)
					if (KNN.coeff(it.row(), j) == 1)
						++n_knn;
				distI[id][j] = (double)n_knn / snn;
				//printf("%d\t%d\t%lf\n", n_knn, snn, distI[id][j]);
			}
			topKIndex(distI[id], n_data, distWithIndex[id], kNeighborNumber + 1, 0, 1);
			for (j = 0; j <= kNeighborNumber; ++j)
			{
				buffers[id].push_back(Triplet<double>(i, distWithIndex[id][j], 1));
				buffers[id].push_back(Triplet<double>(distWithIndex[id][j], i, 1));
			}
		}

#pragma omp master
		{
			for (auto & buffer : buffers)
			{
				move(buffer.begin(), buffer.end(), back_inserter(coefficients));
			}
			for (i = 0; i < n_threads; ++i)
				free(distI[i]);
			free(distI);
			for (i = 0; i < n_threads; ++i)
				free(distWithIndex[i]);
			free(distWithIndex);
			//printf("End parallel\n");
		}
	}

//#pragma omp parallel for schedule(dynamic, 8) private(j)
//	for (i = 0; i < KNN.outerSize(); ++i)
//	{
//		if (i % 1000 == 0)
//			printf("L2: %d\n", i);
//		auto id = omp_get_thread_num();
//		for (j = i + 1; j < KNN.outerSize(); ++j)
//		{
//			int n_knn = 0;
//			for (SparseMatrix<double>::InnerIterator it(KNN, i); it; ++it)
//				if (KNN.coeff(it.row(), j) == 1)
//					++n_knn;
//			if (((double)n_knn / snn) >= simcos_threshold)
//			{
//				//printf("%d\t%d: %f\t%f\n", i, j, ((double)n_knn / snn), simcos_threshold);
//				buffers[id].push_back(Triplet<double>(i, j, 1));
//				buffers[id].push_back(Triplet<double>(j, i, 1));
//			}
//		}
//	}
//
//	for (auto & buffer : buffers)
//	{
//		move(buffer.begin(), buffer.end(), back_inserter(coefficients));
//	}
	L.setFromTriplets(coefficients.begin(), coefficients.end());

	/*if (err = fopen_s(&fp, "L1.txt", "w"))
	{
		printf("timeFile error value: %d", err);
		exit(1);
	}
	for (i = 0; i < L.rows(); ++i)
	{
		for (j = 0; j < L.cols(); ++j)
			fprintf(fp, "%.18f\t", L.coeff(i, j));
		fprintf(fp, "\n");
	}
	fclose(fp);*/

	for (i = 0; i < L.outerSize(); ++i)
	{
		degree = 0;
		for (SparseMatrix<double>::InnerIterator it(L, i); it; ++it)
			if (it.value() >= 2)
			{
				//printf("%d\t%d\t%f\t", it.col(), it.row(), it.value());
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
	L.prune(1, 1e-8);
	i = 0;
	n_connectedComponent = 0;
	top = -1;
	int *dataConnectedComponentIndex = (int *)calloc(n_data, sizeof(int));
	int *visit = (int *)calloc(n_data, sizeof(int));
	int *stack = (int *)calloc(n_data, sizeof(int));
	while (1)
	{
		while (i < n_data && visit[i])
			++i;
		if (i == n_data)
			break;
		visit[i] = 1;
		stack[++top] = i;
		dataConnectedComponentIndex[i] = n_connectedComponent;
		while (top > -1)
		{
			j = stack[top--];
			for (SparseMatrix<double>::InnerIterator it(L, j); it; ++it)
			{
				if (it.value() == -1 && visit[it.row()] == 0)
				{
					visit[it.row()] = 1;
					stack[++top] = it.row();
					dataConnectedComponentIndex[it.row()] = n_connectedComponent;
				}
			}
		}
		++n_connectedComponent;
	}

	free(stack);
	free(visit);
	//printf("the number of connected component of mutual knn graph: %d\n", n_connectedComponent);
	if (n_connectedComponent == 1)
		return L;
	int *connectedComponentIndex = (int *)calloc(n_connectedComponent, sizeof(int));
	L2WithIndex *distP = (L2WithIndex *)malloc(n_connectedComponent * (n_connectedComponent - 1) / 2 * sizeof(L2WithIndex));

	//construct kdtree
	vector<vector<KDTreePoint>> component_point(n_connectedComponent);
	vector<vector<int>> component_point_id(n_connectedComponent);
	for (int i = 0; i < n_data; ++i)
	{
		int id = dataConnectedComponentIndex[i];
		component_point[id].push_back(KDTreePoint(dataMat[i]));
		component_point_id[id].push_back(i);
	}
	vector<KDTree<KDTreePoint>> component_kdtree(component_point.begin(), component_point.end());

#pragma omp parallel for schedule(dynamic, 8) private(j)
	for (i = 0; i < n_connectedComponent; ++i)
	{
		/*if (i % 100 == 0)
			printf("L3: %d\n", i);*/
		connectedComponentIndex[i] = i;

		for (j = i + 1; j < n_connectedComponent; ++j)
		{
			L2WithIndex minDist;
			minDist.dist = DBL_MAX;
			minDist.row = -1;
			minDist.col = -1;

			for (size_t k = 0; k < component_point[i].size(); ++k)
			{
				vector<int> ids = component_kdtree[j].knnSearch(component_point[i][k], 20);
				double tempDist = n_data;
				int id = -1;
				for (size_t l = 0; l < ids.size(); ++l)
				{
					double d = disKNN(KNN, component_point_id[i][k], component_point_id[j][ids[l]]);
					d = n_data - d;
					if (d < tempDist) {
						tempDist = d;
						id = ids[l];
					}
				}
				/*for (int l = 0; l < n_dimension; ++l)
					tempDist += pow(component_point[i][k][l] - component_point[j][id][l], 2);*/
				if (tempDist < minDist.dist)
				{
					minDist.dist = tempDist;
					minDist.row = component_point_id[i][k];
					minDist.col = component_point_id[j][id];
				}
			}
			//minDist.dist = sqrt(minDist.dist);

			int index = (n_connectedComponent - 1) * i - (i + i * i) / 2 + j - 1;
			distP[index] = minDist;
		}

		/*for (j = i + 1; j < n_connectedComponent; ++j)
		{
			L2WithIndex minDist;
			minDist.dist = DBL_MAX;
			minDist.row = -1;
			minDist.col = -1;

			for (size_t k = 0; k < component_point[i].size(); ++k)
			{
				int id = component_kdtree[j].nnSearch(component_point[i][k]);
				double tempDist = 0;
				for (int l = 0; l < n_dimension; ++l)
					tempDist += pow(component_point[i][k][l] - component_point[j][id][l], 2);
				if (tempDist < minDist.dist)
				{
					minDist.dist = tempDist;
					minDist.row = component_point_id[i][k];
					minDist.col = component_point_id[j][id];
				}
			}
			minDist.dist = sqrt(minDist.dist);

			int index = (n_connectedComponent - 1) * i - (i + i * i) / 2 + j - 1;
			distP[index] = minDist;
		}*/
	}
	sort(distP, distP + (n_connectedComponent * (n_connectedComponent - 1) / 2), cmp);
	//printf("sort finish\n");
	interConnectedComponentEdge = 0;
	int minDistIndex = 0, mergeLabel1, mergeLabel2;
	while (interConnectedComponentEdge < n_connectedComponent - 1)
	{
		while (connectedComponentIndex[dataConnectedComponentIndex[distP[minDistIndex].row]] ==
			connectedComponentIndex[dataConnectedComponentIndex[distP[minDistIndex].col]])
			++minDistIndex;
		if (L.coeffRef(distP[minDistIndex].row, distP[minDistIndex].col) != 0 ||
			L.coeffRef(distP[minDistIndex].col, distP[minDistIndex].row) != 0)
		{
			printf("L(%d, %d) = %f", distP[minDistIndex].row, distP[minDistIndex].col,
				L.coeffRef(distP[minDistIndex].row, distP[minDistIndex].col));
			exit(0);
		}
		L.coeffRef(distP[minDistIndex].row, distP[minDistIndex].col) = -1;
		L.coeffRef(distP[minDistIndex].col, distP[minDistIndex].row) = -1;
		L.coeffRef(distP[minDistIndex].row, distP[minDistIndex].row) += 1;
		L.coeffRef(distP[minDistIndex].col, distP[minDistIndex].col) += 1;
		mergeLabel1 = connectedComponentIndex[dataConnectedComponentIndex[distP[minDistIndex].row]];
		mergeLabel2 = connectedComponentIndex[dataConnectedComponentIndex[distP[minDistIndex].col]];
		for (i = 0; i < n_connectedComponent; ++i)
			if (connectedComponentIndex[i] == mergeLabel1)
				connectedComponentIndex[i] = mergeLabel2;
		++interConnectedComponentEdge;
	}
	free(dataConnectedComponentIndex);
	free(connectedComponentIndex);
	free(distP);
	L.prune(1, 1e-8);
	return L;
}

SparseMatrix<double> calcLaplaceMat()
{
	int i, j, k, degree;
	vector<Triplet<double>> coefficients;
	vector<Triplet<double>> coefficientsCol;
	vector<vector<Triplet<double> > > buffers;
	vector<vector<Triplet<double> > > buffersCol;
	/*vector<vector<double> > distI(12, vector<double>(n_data));
	vector<vector<int> > distWithIndex(12, vector<int>(n_data));*/
	SparseMatrix<double> L(n_data, n_data);
	SparseMatrix<double> Lcol(n_data, n_data);
	double **distI;
	int **distWithIndex;
	int n_threads;

#pragma omp parallel
	{
#pragma omp single
		{
			n_threads = omp_get_max_threads();
			distI = (double **)malloc(n_threads * sizeof(double *));
			for (i = 0; i < n_threads; ++i)
				distI[i] = (double *)malloc(n_data * sizeof(double));
			distWithIndex = (int **)malloc(n_threads * sizeof(int *));
			for (i = 0; i < n_threads; ++i)
				distWithIndex[i] = (int *)malloc((kNeighborNumber + 1) * sizeof(int));
			/*vector<vector<double> > distI(n_threads, vector<double>(n_data));
			vector<vector<int> > distWithIndex(n_threads, vector<int>(n_data));*/
			//#pragma omp master
			buffers.resize(n_threads);
			buffersCol.resize(n_threads);
			//distI = vector<vector<double> >(n_threads, vector<double>(n_data));
			//distWithIndex = vector<vector<int> >(n_threads, vector<int>(n_data));
			printf("Start parallel: %d threads\n", n_threads);
		}
#pragma omp for private(j) private(k)
		for (i = 0; i < n_data; ++i)
		{
			auto id = omp_get_thread_num();
			for (j = 0; j < n_data; ++j)
			{
				double distIJ = 0;
				for (k = 0; k < n_dimension; ++k)
					distIJ += pow(dataMat[i][k] - dataMat[j][k], 2);
				distIJ = pow(distIJ, 0.5);
				distI[id][j] = distIJ;
			}
			/*argsort(distI[id], distWithIndex[id], 0);
			for (j = 0; j <= kNeighborNumber; ++j)
			{
				buffers[id].push_back(Triplet<double>(i, distWithIndex[id][j], 1));
				buffersCol[id].push_back(Triplet<double>(distWithIndex[id][j], i, 1));
			}*/
			topKIndex(distI[id], n_data, distWithIndex[id], kNeighborNumber + 1, 0, 0);
			for (j = 0; j <= kNeighborNumber; ++j)
			{
				buffers[id].push_back(Triplet<double>(i, distWithIndex[id][j], 1));
				buffersCol[id].push_back(Triplet<double>(distWithIndex[id][j], i, 1));
			}
		}
		
#pragma omp master
		{
			for (auto & buffer : buffers)
			{
				move(buffer.begin(), buffer.end(), back_inserter(coefficients));
			}
			for (auto & bufferCol : buffersCol)
			{
				move(bufferCol.begin(), bufferCol.end(), back_inserter(coefficientsCol));
			}
			for (i = 0; i < n_threads; ++i)
				free(distI[i]);
			free(distI);
			for (i = 0; i < n_threads; ++i)
				free(distWithIndex[i]);
			free(distWithIndex);
			printf("End parallel\n");
		}
	}
	L.setFromTriplets(coefficients.begin(), coefficients.end());
	Lcol.setFromTriplets(coefficientsCol.begin(), coefficientsCol.end());
	L += Lcol;
	for (i = 0; i < L.outerSize(); ++i)
	{
		degree = 0;
		/*for (SparseMatrix<double>::InnerIterator it(L, i); it; ++it)
			if (it.value() >= 2)
			{
				it.valueRef() = -1;
				++degree;
			}
			else
			{
				it.valueRef() = 0;
			}*/
		for (SparseMatrix<double>::InnerIterator it(L, i); it; ++it)
			if (it.value() != 0)
			{
				it.valueRef() = -1;
				++degree;
			}
		L.coeffRef(i, i) = degree - 1;
	}
	/*for (SparseMatrix<double>::InnerIterator it(L, 0); it; ++it)
		it.valueRef() = 0;
	L.coeffRef(0, 0) = 1;
	for (i = 1; i < L.outerSize(); ++i)
	{
		degree = 0;
		for (SparseMatrix<double>::InnerIterator it(L, i); it; ++it)
			if (it.value() != 0)
			{
				++degree;
				if(it.col() == 0)
					it.valueRef() = 0;
				else
					it.valueRef() = -1;
			}
		L.coeffRef(i, i) = degree - 1;
	}*/
	L.makeCompressed();
	//L.prune(1, 1e-8);
	return L;
}

bool cmp(const L2WithIndex a, const L2WithIndex b)
{
	if (a.dist != b.dist)
		return a.dist < b.dist;
	else if (a.row != b.row)
		return a.row < b.row;
	else
		return a.col < b.col;
}

SparseMatrix<double> constructLaplaceMat()
{
	int i, j, k, index, n_edge = 0, minDistIndex = 0, mergeLabel1, mergeLabel2, degree;
	vector<Triplet<double>> coefficients;
	SparseMatrix<double> L(n_data, n_data);
	int *label = (int *)calloc(n_data, sizeof(int));
	L2WithIndex *distP = (L2WithIndex *)malloc(n_data * (n_data - 1) / 2 * sizeof(L2WithIndex));
	for (i = 0; i < n_data; ++i)
	{
		label[i] = i;
		for (j = i + 1; j < n_data; ++j)
		{
			index = (n_data - 1) * i - (i + i * i) / 2 + j - 1;
			distP[index].row = i;
			distP[index].col = j;
			distP[index].dist = 0;
			for (k = 0; k < n_dimension; ++k)
				distP[index].dist += pow(dataMat[i][k] - dataMat[j][k], 2);
			distP[index].dist = pow(distP[index].dist, 0.5);
		}
	}
	/*for (i = 0; i < n_data * (n_data - 1) / 2; ++i)
		printf("%f ", dist[i].dist);
	printf("\n\n");*/
	sort(distP, distP + n_data * (n_data - 1) / 2, cmp);
	printf("%f\n", distP[0].dist);
	/*for (i = 0; i < n_data * (n_data - 1) / 2; ++i)
		printf("%f ", dist[i].dist);
	printf("\n\n");*/
	while (n_edge < n_data - 1)
	{
		/*printf("%d:\n", n_edge);
		for (i = 0; i < n_data; ++i)
			printf("%d ", label[i]);
		printf("\n");*/
		while (label[distP[minDistIndex].row] == label[distP[minDistIndex].col])
			++minDistIndex;
		coefficients.push_back(Triplet<double>(distP[minDistIndex].row, distP[minDistIndex].col, 1));
		coefficients.push_back(Triplet<double>(distP[minDistIndex].col, distP[minDistIndex].row, 1));
		//printf("%d\t%d\n", dist[minDistIndex].row, dist[minDistIndex].col);
		/*mergeLabel1 = label[dist[minDistIndex].row] > label[dist[minDistIndex].col] ? 
			label[dist[minDistIndex].row] : label[dist[minDistIndex].col];
		mergeLabel2 = label[dist[minDistIndex].row] <= label[dist[minDistIndex].col] ?
			label[dist[minDistIndex].row] : label[dist[minDistIndex].col];*/
		mergeLabel1 = label[distP[minDistIndex].row];
		mergeLabel2 = label[distP[minDistIndex].col];
		for (i = 0; i < n_data; ++i)
			if (label[i] == mergeLabel1)
				label[i] = mergeLabel2;
		++n_edge;
	}
	L.setFromTriplets(coefficients.begin(), coefficients.end());
	for (i = 0; i < L.outerSize(); ++i)
	{
		degree = 0;
		for (SparseMatrix<double>::InnerIterator it(L, i); it; ++it)
			if (it.value() != 0)
			{
				it.valueRef() = -1;
				++degree;
			}
		L.coeffRef(i, i) = degree;
	}
	L.makeCompressed();
	free(distP);
	free(label);
	return L;
}

int connectedComponent(const SparseMatrix<double> &Laplace)
{
	int i = 0, j, n_connctedComponent = 0, top = -1;
	int *visit = (int *)calloc(n_data, sizeof(int));
	int *stack = (int *)calloc(n_data, sizeof(int));
	while (1)
	{
		while (i < n_data && visit[i])
			++i;
		if (i == n_data)
			break;
		++n_connctedComponent;
		visit[i] = 1;
		stack[++top] = i;
		while (top > -1)
		{
			j = stack[top--];
			for (SparseMatrix<double>::InnerIterator it(Laplace, j); it; ++it)
			{
				if (it.value() == -1 && visit[it.row()] == 0)
				{
					visit[it.row()] = 1;
					stack[++top] = it.row();
				}
			}
		}
	}
	free(stack);
	free(visit);
	return n_connctedComponent;
}

//SparseMatrix<double, RowMajor> calcGreenFunctionGradient(const SparseMatrix<double> &Laplace)
//{
//	int i, j, k, n_threads;
//	double **GradPerEdge;
//	int **GradPerEdgeIndex;
//	//clock_t t1;
//
//	//MatrixXd g(n_data, n_data);
//	//vector<VectorXd> g(n_data);
//	VectorXd mean;
//	SparseMatrix<double> L(n_data, n_data);
//	int n_edge = (int)((Laplace.nonZeros() - n_data) / 2);
//	/*double *dist;
//	dist = (double *)malloc(n_edge * sizeof(double));*/
//	vector<double> dist(n_edge, 0);
//	j = 0;
//	for (i = 0; i < Laplace.outerSize(); ++i)
//		for (SparseMatrix<double>::InnerIterator it(Laplace, i); it; ++it)
//		{
//			if (it.row() < it.col())
//			{
//				for (k = 0; k < n_dimension; ++k)
//					dist[j] += pow((dataMat[it.row()][k] - dataMat[it.col()][k]), 2);
//				dist[j] = pow(dist[j], 0.5);
//				++j;
//			}
//			else
//				break;
//		}
//	//printf("%d\n", j);
//	int largestNEdge = (int)(pow(kNeighborNumber, 2) * pow(n_cluster, 2)) < n_edge ?
//		(int)(pow(kNeighborNumber, 2) * pow(n_cluster, 2)) : n_edge;
//	//int largestNEdge = 100;
//	//int largestNEdge = int(n_edge * 0.01);
//	//int largestNEdge = n_edge;
//	vector<Triplet<double> > coefficients;
//	vector<vector<Triplet<double> > > buffers;
//	/*VectorXi prealloc(n_data);
//	prealloc.fill(largestNEdge);*/
//	SparseMatrix<double> gradient(n_data, n_edge);
//	printf("%d rows, %d cols\n", n_data, n_edge);
//	//gradient.reserve(prealloc);
//	L = Laplace;
//	for (SparseMatrix<double>::InnerIterator it(L, 0); it; ++it)
//		if (it.value() != 0)
//			L.coeffRef(it.row(), it.col()) = L.coeffRef(it.col(), it.row()) = 0;
//	L.coeffRef(0, 0) = 1;
//	SimplicialLDLT<SparseMatrix<double> > solver;
//	//t1 = clock();
//	solver.compute(L);
//	//printf("computer:%f\n", (double)(clock() - t1) / CLOCKS_PER_SEC);
//	//t1 = clock();
//#pragma omp parallel
//	{
//#pragma omp single
//		{
//			n_threads = omp_get_max_threads();
//			GradPerEdge = (double **)malloc(n_threads * sizeof(double *));
//			for (i = 0; i < n_threads; ++i)
//				GradPerEdge[i] = (double *)malloc(n_edge * sizeof(double));
//			GradPerEdgeIndex = (int **)malloc(n_threads * sizeof(int *));
//			for (i = 0; i < n_threads; ++i)
//				GradPerEdgeIndex[i] = (int *)malloc(largestNEdge * sizeof(int));
//			//#pragma omp master
//			//auto n_threads = omp_get_max_threads();
//			buffers.resize(n_threads);
//			printf("Start parallel: %d threads\n", n_threads);
//		}
//#pragma omp for private(j)
//		for (i = 0; i < n_data; ++i)
//		{
//			int count = 0;
//			auto id = omp_get_thread_num();
//			/*vector<double> GradPerEdge(n_edge);
//			vector<int> GradPerEdgeIndex(n_edge);*/
//			/*double *GradPerEdge = (double *)malloc(n_edge * sizeof(double));
//			int *GradPerEdgeIndex = (int *)malloc(n_edge * sizeof(int));*/
//			VectorXd b(n_data);
//			VectorXd sub(n_data);
//			VectorXd gi(n_data);
//			b.fill(-(double)1 / n_data);
//			b[i] += 1;
//			b[0] = 0;
//			//g.col(i) = solver.solve(b);
//			gi = solver.solve(b);
//			sub.fill(gi.mean());
//			gi -= sub;
//			//printf("%d:%f\n", i, gi[0]);
//			/*if (i % 1000 == 0)
//				printf("grad: %d\n", i);*/
//			for (j = 0; j < Laplace.outerSize(); ++j)
//				for (SparseMatrix<double>::InnerIterator it(Laplace, j); it; ++it)
//				{
//					if (it.row() < it.col())
//					{
//						/*double dist = 0;
//						for (k = 0; k < n_dimension; ++k)
//							dist += pow((dataMat[it.row()][k] - dataMat[it.col()][k]), 2);
//						dist = pow(dist, 0.5);
//						GradPerEdge[id][count++] = (gi[it.col()] - gi[it.row()]) * dist;*/
//						GradPerEdge[id][count] = (gi[it.col()] - gi[it.row()]) * dist[count];
//						++count;
//					}
//					else
//						break;
//				}
//			//cout << count << endl;
//			//argsort(GradPerEdge[id], GradPerEdgeIndex[id], 1);
//			topKIndex(GradPerEdge[id], n_edge, GradPerEdgeIndex[id], largestNEdge, 1, 1);
//			for (j = 0; j < largestNEdge; ++j)
//				buffers[id].push_back(Triplet<double>(i, GradPerEdgeIndex[id][j], GradPerEdge[id][GradPerEdgeIndex[id][j]]));
//		}
//#pragma omp master
//		{
//			for (auto & buffer : buffers)
//			{
//				move(buffer.begin(), buffer.end(), back_inserter(coefficients));
//			}
//			for (i = 0; i < n_threads; ++i)
//				free(GradPerEdge[i]);
//			free(GradPerEdge);
//			for (i = 0; i < n_threads; ++i)
//				free(GradPerEdgeIndex[i]);
//			free(GradPerEdgeIndex);
//			printf("End parallel\n");
//		}
//	}
//	gradient.setFromTriplets(coefficients.begin(), coefficients.end());
//	gradient.makeCompressed();
//
//	vector<double> dim_val(gradient.cols(), 0);
//	for (int i = 0; i < gradient.cols(); ++i)
//		for (SparseMatrix<double>::InnerIterator it(gradient, i); it; ++it)
//			dim_val[it.col()] += abs(it.value());
//	/*double sum1 = 0, sum2 = 0;
//	for (int i = 0; i < gradient.cols(); ++i)
//	{
//		for (SparseMatrix<double>::InnerIterator it(gradient, i); it; ++it)
//		{
//			sum1 += pow(it.value(), 2.0);
//			sum2 += it.value();
//		}
//		dim_val[i] = sum1/ gradient.rows() - pow(sum2 / gradient.rows(), 2.0);
//	}*/
//
//	/*vector<double> val = dim_val;
//	sort(val.begin(), val.end());
//	for (size_t i = 0; i < val.size(); ++i)
//		cout << val[i] << " ";
//	cout << endl;*/
//
//	size_t reserved_ele = static_cast<size_t>(largestNEdge);
//	priority_queue<double, vector<double>, greater<double>> heap;
//	for (size_t i = 0; i < dim_val.size(); ++i)
//	{
//		if (heap.size() >= reserved_ele)
//		{
//			if (heap.top() < dim_val[i])
//			{
//				heap.pop();
//				heap.push(dim_val[i]);
//			}
//		}
//		else
//			heap.push(dim_val[i]);
//	}
//	double threshold = heap.top();
//	cout << "threshold " << threshold << endl;
//	vector<int> indices;
//	for (size_t i = 0; i < dim_val.size(); ++i)
//		if (dim_val[i] > threshold) indices.push_back(i);
//
//	coefficients.clear();
//	for (size_t i = 0; i < indices.size(); ++i)
//	{
//		int col = indices[i];
//		for (SparseMatrix<double>::InnerIterator it(gradient, col); it; ++it)
//			coefficients.push_back(Eigen::Triplet<double>(it.row(), i, it.value()));
//	}
//	SparseMatrix<double, RowMajor> new_gradient(gradient.rows(), indices.size());
//	new_gradient.setFromTriplets(coefficients.begin(), coefficients.end());
//
//	//	SparseMatrix<double, RowMajor> new_gradient(gradient);
//
//	return new_gradient;
//	//return gradient;
//}

SparseMatrix<double, RowMajor> calcGreenFunctionGradient(const SparseMatrix<double> &Laplace)
{
	int i, j, k, n_threads;
	double **GradPerEdge;
	int **GradPerEdgeIndex;
	//clock_t t1;

	//MatrixXd g(n_data, n_data);
	//vector<VectorXd> g(n_data);
	SparseMatrix<double> L(n_data, n_data);

	SparseMatrix<double> KNN(n_data, n_data);
	vector<Triplet<double>> coefficients;
	vector<vector<Triplet<double> > > buffers;

	n_threads = omp_get_max_threads();
	buffers.resize(n_threads);
	//printf("Start parallel: %d threads\n", n_threads);

	vector<KDTreePoint> points(dataMat.begin(), dataMat.end());
	KDTree<KDTreePoint> kdtree(points);
	KDTreePoint::DIM = n_dimension;

#pragma omp parallel for
	for (int i = 0; i < n_data; ++i)
	{
		/*if (i % 1000 == 0)
			printf("L1: %d\n", i);*/

		KDTreePoint p(dataMat[i]);
		vector<int> knn_ind = kdtree.knnSearch(p, snn + 1);
		auto id = omp_get_thread_num();
		for (size_t j = 0; j < knn_ind.size(); ++j)
		{
			/*if (labels[i] != labels[knn_ind[j]] && rand() < RAND_MAX * 0.95)
				continue;*/
				//buffers[id].push_back(Triplet<double>(i, knn_ind[j], 1));
			if (knn_ind[j] != i)
				buffers[id].push_back(Triplet<double>(knn_ind[j], i, 1));
		}
	}

	for (auto & buffer : buffers)
	{
		move(buffer.begin(), buffer.end(), back_inserter(coefficients));
	}
	KNN.setFromTriplets(coefficients.begin(), coefficients.end());
	//printf("%d\n", KNN.nonZeros());
	coefficients.clear();
	for (i = 0; i < n_threads; ++i)
		buffers[i].clear();

	int n_edge = (int)((Laplace.nonZeros() - n_data) / 2);
	/*double *dist;
	dist = (double *)malloc(n_edge * sizeof(double));*/
	vector<double> dist(n_edge, 0);
	j = 0;
	for (i = 0; i < Laplace.outerSize(); ++i)
		for (SparseMatrix<double>::InnerIterator it(Laplace, i); it; ++it)
		{
			if (it.row() < it.col())
			{
				int n_knn = 0;
				for (SparseMatrix<double>::InnerIterator it0(KNN, it.row()); it0; ++it0)
					if (KNN.coeff(it0.row(), it.col()) == 1)
						++n_knn;
				dist[j] = 1 - (double)n_knn / snn;
				/*for (k = 0; k < n_dimension; ++k)
					dist[j] += pow((dataMat[it.row()][k] - dataMat[it.col()][k]), 2);
				dist[j] = pow(dist[j], 0.5);*/
				++j;
			}
			else
				break;
		}
	//printf("%d\n", j);
	int largestNEdge = min((int)(pow(kNeighborNumber, 1) * pow(n_cluster, 2)), n_edge);
	//int largestNEdge = 100;
	//int largestNEdge = int(n_edge * 0.01);
	//int largestNEdge = n_edge;
	/*vector<Triplet<double> > coefficients;
	vector<vector<Triplet<double> > > buffers;*/
	/*VectorXi prealloc(n_data);
	prealloc.fill(largestNEdge);*/
	printf("%d rows, %d cols\n", n_data, n_edge);
//	gradient.reserve(prealloc);
	L = Laplace;
	for (SparseMatrix<double>::InnerIterator it(L, 0); it; ++it)
		if (it.value() != 0)
			L.coeffRef(it.row(), it.col()) = L.coeffRef(it.col(), it.row()) = 0;
	L.coeffRef(0, 0) = 1;
	SimplicialLDLT<SparseMatrix<double> > solver;
	//t1 = clock();
	solver.compute(L);
	//printf("computer:%f\n", (double)(clock() - t1) / CLOCKS_PER_SEC);
	//t1 = clock();

	srand(0);
	size_t sample_n = min(1000, n_data / 10);
	VectorXd sum(n_edge);
	sum.setZero();

	vector<pair<int, int>> edges_id(n_edge);
	vector<int> samples;
	while(samples.size() < sample_n)
	{
		int id = rand() % n_data;
		if (find(samples.begin(), samples.end(), id) == samples.end())
			samples.push_back(id);
		else
			continue;

		VectorXd b(n_data);
		VectorXd sub(n_data);
		b.fill(-(double)1 / n_data);
		b[id] += 1;
		b[0] = 0;
		VectorXd gi = solver.solve(b);
		sub.fill(gi.mean());
		gi -= sub;
		int count = 0;

		VectorXd grad(n_edge);
		for (j = 0; j < Laplace.outerSize(); ++j)
		{
			for (SparseMatrix<double>::InnerIterator it(Laplace, j); it; ++it)
			{
				if (it.row() < it.col())
				{
					grad(count) = abs((gi[it.col()] - gi[it.row()]) * dist[count]);
					edges_id[count] = make_pair(it.col(), it.row());
					++count;
				}
				else
					break;
			}
		}
		sum += grad;
	}

	size_t reserved_ele = static_cast<size_t>(largestNEdge);
	auto comp = [](const pair<double, int>& e1, const pair<double, int>& e2){
		return e1.first > e2.first;
	};
	priority_queue<pair<double, int>, vector<pair<double, int>>, decltype(comp)> heap(comp);
	for (int i = 0; i < sum.rows(); ++i)
	{
		if (heap.size() >= reserved_ele)
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

	/*for (size_t i = 0; i < sum.rows(); ++i)
		cout << sum[i] << " ";
	cout << endl;
	cout << "threshold " << heap.top().first << endl;*/

	vector<int> indices;
	while (!heap.empty())
	{
		indices.push_back(heap.top().second);
		heap.pop();
	}
	sort(indices.begin(), indices.end());

	//FILE *fp;
	//errno_t err;
	//int *labels = (int *)calloc(n_data, sizeof(int));
	//if (err = fopen_s(&fp, "complexStructureLabels.txt", "r"))
	//{
	//	printf("timeFile error value: %d", err);
	//	exit(1);
	//}
	//for (i = 0; i < n_data; ++i)
	//	fscanf_s(fp, "%d", &labels[i]);
	//fclose(fp);

	//int interClassEdges = 0;

	//for (size_t i = 0; i < indices.size(); ++i)
	//	if (labels[edges_id[indices[i]].first] != labels[edges_id[indices[i]].second])
	//	{
	//		++interClassEdges;
	//		//printf("%d\t%d\n", edges_id[indices[i]].first, edges_id[indices[i]].second);
	//		printf("%f\t\n", sum[indices[i]]);
	//	}
	//printf("\n");
	//for (size_t i = 0; i < 20 && i < largestNEdge; ++i)
	//	if (labels[edges_id[indices[i]].first] == labels[edges_id[indices[i]].second])
	//	{
	//		//printf("%d\t%d\n", edges_id[indices[i]].first, edges_id[indices[i]].second);
	//		printf("%f\t\n", sum[indices[i]]);
	//	}
	//printf("%d条边， %d类间边\n", largestNEdge,	interClassEdges);

#pragma omp parallel
	{
#pragma omp single
		{
			n_threads = omp_get_max_threads();
			GradPerEdge = (double **)malloc(n_threads * sizeof(double *));
			for (i = 0; i < n_threads; ++i)
				GradPerEdge[i] = (double *)malloc(largestNEdge * sizeof(double));
			//#pragma omp master
			//auto n_threads = omp_get_max_threads();
			buffers.resize(n_threads);
			//printf("Start parallel: %d threads\n", n_threads);
		}
#pragma omp for private(j)
		for (i = 0; i < n_data; ++i)
		{
			int count = 0;
			auto id = omp_get_thread_num();
			/*vector<double> GradPerEdge(n_edge);
			vector<int> GradPerEdgeIndex(n_edge);*/
			/*double *GradPerEdge = (double *)malloc(n_edge * sizeof(double));
			int *GradPerEdgeIndex = (int *)malloc(n_edge * sizeof(int));*/
			VectorXd b(n_data);
			VectorXd sub(n_data);
			VectorXd gi(n_data);
			b.fill(-(double)1 / n_data);
			b[i] += 1;
			b[0] = 0;
			//g.col(i) = solver.solve(b);
			gi = solver.solve(b);
			sub.fill(gi.mean());
			gi -= sub;

			for (size_t j = 0; j < indices.size(); ++j)
				GradPerEdge[id][j] = (gi[edges_id[indices[j]].first] - gi[edges_id[indices[j]].second])*dist[indices[j]];

			for (j = 0; j < largestNEdge; ++j)
				buffers[id].push_back(Triplet<double>(i, j, GradPerEdge[id][j]));
		}
#pragma omp master
		{
			for (auto & buffer : buffers)
			{
				move(buffer.begin(), buffer.end(), back_inserter(coefficients));
			}
			for (i = 0; i < n_threads; ++i)
				free(GradPerEdge[i]);
			free(GradPerEdge);
			//printf("End parallel\n");
		}
	}
	SparseMatrix<double, RowMajor> gradient(n_data, indices.size());
	gradient.setFromTriplets(coefficients.begin(), coefficients.end());
	gradient.makeCompressed();
	return gradient;
}

//SparseMatrix<double, RowMajor> getGradient(const SparseMatrix<double> &Laplace, const vector<VectorXd> &g, double ratio)
//{
//	int i, j, k;
//	
//	int n_edge = (int)((Laplace.nonZeros() - n_data) / 2);
//	//cout << n_edge << endl;
//	
//	int largestNEdge = (int)(n_edge * ratio);
//	vector<Triplet<double> > coefficients;
//	vector<vector<Triplet<double> > > buffers;
//	VectorXi prealloc(n_data);
//	prealloc.fill(largestNEdge);
//	SparseMatrix<double, RowMajor> gradient(n_data, n_edge);
//	gradient.reserve(prealloc);
//
//#pragma omp parallel
//	{
//#pragma omp master
//		{
//			auto n_threads = omp_get_max_threads();
//			buffers.resize(n_threads);
//		}
//#pragma omp for private(j) private(k)
//			for (i = 0; i < n_data; ++i)
//			{
//				auto id = omp_get_thread_num();
//				int count = 0;
//				double *GradPerEdge = (double *)malloc(n_edge * sizeof(double));
//				int *GradPerEdgeIndex = (int *)malloc(n_edge * sizeof(int));
//				for (j = 0; j < Laplace.outerSize(); ++j)
//					for (SparseMatrix<double>::InnerIterator it(Laplace, j); it; ++it)
//						if (it.row() > it.col()/* && it.value() == -1*/)
//						{
//							double dist = 0;
//							for (k = 0; k < n_dimension; ++k)
//								dist += pow((dataMat[it.row()][k] - dataMat[it.col()][k]), 2);
//							dist = pow(dist, 0.5);
//							GradPerEdge[count++] = (g[i][it.col()] - g[i][it.row()]) * dist;
//						}
//				//cout << count << endl;
//				argsort(GradPerEdge, GradPerEdgeIndex, n_edge, 1);
//				for (j = n_edge - 1; j >= n_edge - largestNEdge; --j)
//					buffers[id].push_back(Triplet<double>(i, GradPerEdgeIndex[j], GradPerEdge[GradPerEdgeIndex[j]]));
//				free(GradPerEdge);
//				free(GradPerEdgeIndex);
//		}
//
//#pragma omp master
//		{
//			for (auto & buffer : buffers)
//			{
//				move(buffer.begin(), buffer.end(), back_inserter(coefficients));
//			}
//		}
//	}
//	gradient.setFromTriplets(coefficients.begin(), coefficients.end());
//	gradient.makeCompressed();
//	return gradient;
//}

void kMeansPlusPlusInitWithCSRMat(double **centroids, const SparseMatrix<double, RowMajor> &data, int n_clusters)
{
	if (n_clusters < 1)
		return;
	int i, j, n_col = data.cols(), n_row = data.rows(), n_cent;
	double distSum, randomProb;
	double *distCent = (double *)calloc(n_clusters, sizeof(double));
	//vector<double> distCent(n_cent, 0);
	vector<double> dist(n_row);
	srand((int)time(NULL) * 100000000);
	//srand(11);
	i = (int)(((double)rand() / RAND_MAX) * n_row);
	for (SparseMatrix<double, RowMajor>::InnerIterator it(data, i); it; ++it)
	{
		centroids[0][it.col()] = it.value();
		distCent[0] += pow(it.value(), 2);
	}
	n_cent = 1;
	while (n_cent < n_clusters)
	{
		for (i = 0; i < n_row; ++i)
			dist[i] = DBL_MAX;
#pragma omp parallel for private(j)
		for (i = 0; i < n_row; ++i)
			for (j = 0; j < n_cent; ++j)
			{
				double distIJ = distCent[j];
				for(SparseMatrix<double, RowMajor>::InnerIterator it(data, i); it; ++it)
					distIJ = distIJ - pow(centroids[j][it.col()], 2) + pow((it.value() - centroids[j][it.col()]), 2);
				if (distIJ < dist[i])
					dist[i] = distIJ;
			}
		distSum = 0;
		for (i = 0; i < n_row; ++i)
			distSum += dist[i];
		for (i = 0; i < n_row; ++i)
			dist[i] /= distSum;
		for (i = 1; i < n_row; ++i)
			dist[i] += dist[i - 1];
		randomProb = (double)rand() / RAND_MAX;
		for (i = 0; i < n_row; ++i)
			if (randomProb < dist[i])
			{
				for (SparseMatrix<double, RowMajor>::InnerIterator it(data, i); it; ++it)
				{
					centroids[n_cent][it.col()] = it.value();
					distCent[n_cent] += pow(it.value(), 2);
				}
				break;
			}
		++n_cent;
	}
	free(distCent);;
	printf("init\n");
}

void kMeansWithCSRMat(int *clusterAssessment, const SparseMatrix<double, RowMajor> &data, int n_clusters)
{
	bool clusterChanged;
	int i, j, n_col = data.cols(), n_row = data.rows(), minIndex, changeCount = n_row;
	double **centroids, minDist;
	vector<double> distCent(n_clusters, 0);
	centroids = (double **)malloc(n_clusters * sizeof(double *));
	for (i = 0; i < n_clusters; ++i)
		centroids[i] = (double *)calloc(n_col, sizeof(double));
	//sumOfDataOfCent = (double *)malloc(n_col * sizeof(double));
	clusterChanged = true;
	kMeansPlusPlusInitWithCSRMat(centroids, data, n_clusters);
	while (clusterChanged /*&& changeCount > int(0.01 * n_row)*/)
	{
		clusterChanged = false;
		//changeCount = 0;
		for (i = 0; i < n_clusters; ++i)
		{
			distCent[i] = 0;
			for (j = 0; j < n_col; ++j)
				distCent[i] += pow(centroids[i][j], 2);
		}
#pragma omp parallel for private(j) private(minDist) private(minIndex)
		for (i = 0; i < n_row; ++i)
		{
			minDist = DBL_MAX;
			minIndex = -1;
			for (j = 0; j < n_clusters; ++j)
			{
				double distIJ = distCent[j];
				for (SparseMatrix<double, RowMajor>::InnerIterator it(data, i); it; ++it)
					distIJ = distIJ - pow(centroids[j][it.col()], 2) + pow((it.value() - centroids[j][it.col()]), 2);
				if (distIJ < minDist)
				{
					minDist = distIJ;
					minIndex = j;
				}
			}
			if (clusterAssessment[i] != minIndex)
			{
				clusterChanged = true;
				clusterAssessment[i] = minIndex;
				//changeCount += 1;
			}
		}
#pragma omp parallel for private(j)
		for (i = 0; i < n_clusters; ++i)
		{
			int count = 0;
			vector<double> sumOfDataOfCent(n_col, 0);
			for (j = 0; j < n_row; ++j)
			{
				if (clusterAssessment[j] == i)
				{
					++count;
					for (SparseMatrix<double, RowMajor>::InnerIterator it(data, j); it; ++it)
						sumOfDataOfCent[it.col()] += it.value();
				}
			}
			if (count == 0)
				kMeansPlusPlusInitWithCSRMat(centroids + i, data, 1);
			else
				for (j = 0; j < n_col; ++j)
					centroids[i][j] = sumOfDataOfCent[j] / count;
		}
	}
	for (i = 0; i < n_clusters; ++i)
		free(centroids[i]);
	free(centroids);
	//free(sumOfDataOfCent);
}

void storeOutcome(char *fileName, int *assesment, clock_t t1, clock_t t2, clock_t t3, clock_t t4)
{
	char a[] = { "RunningTime.txt" };
	char b[] = { "Assessment.txt" };
	FILE *fp;
	errno_t err;
	int lenOfFileName = strlen(fileName), i;
	fileName[lenOfFileName - 4] = '\0';
	char *timeFileName = (char *)malloc((lenOfFileName + 16) * sizeof(char));
	char *assessmentFileName = (char *)malloc((lenOfFileName + 15) * sizeof(char));
	sprintf_s(timeFileName, (lenOfFileName + 16) * sizeof(char), "%s%s", fileName, a);
	sprintf_s(assessmentFileName, (lenOfFileName + 15) * sizeof(char), "%s%s", fileName, b);
	if (err = fopen_s(&fp, timeFileName, "w"))
	{
		printf("timeFile error value: %d", err);
		exit(1);
	}
	fprintf(fp, "%f\t%f\t%f\t%f", (double)(t2 - t1) / CLOCKS_PER_SEC, (double)(t3 - t2) / CLOCKS_PER_SEC, (double)(t4 - t3) / CLOCKS_PER_SEC, (double)(t4 - t1) / CLOCKS_PER_SEC);
	fclose(fp);
	if (err = fopen_s(&fp, assessmentFileName, "w"))
	{
		printf("assesmentFile error value: %d", err);
		exit(1);
	}
	for (i = 0; i < n_data; ++i)
		fprintf(fp, "%d\n", assesment[i]);
	fclose(fp);
	/*char *cmd = (char *)malloc((21+ strlen(fileName)+ strlen(assessmentFileName)) * sizeof(char));
	sprintf_s(cmd, (21 + strlen(fileName) + strlen(assessmentFileName)) * sizeof(char), "%s%s.txt %s", "python show.py ", fileName, assessmentFileName);
	printf("%s\n", cmd);
	system(cmd);
	free(cmd);*/
	//system("python reuters10k_acc.py");
	free(timeFileName);
	free(assessmentFileName);
}