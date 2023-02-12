#include "pch.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <random>

#define DLLEXPORT extern "C" __declspec(dllexport)

#pragma region LINEAR MODEL

DLLEXPORT double* linearTraining(double* points, double points_size, double* classification, int tabLenght, int maxIteration, double* w, double w_size) {

	//Init du temps pour le random
	srand(time(NULL));

	//Boucle d'entrainement
	for (int iteration = 0; iteration < maxIteration; iteration++)
	{
		//Choix d'un index aléatoire dans le tableau d'éléments
		int k = rand() % tabLenght;

		//Récupération de sa valeur
		double yk = classification[k];

		//Vecteur d'entrée avec le biais pour éviter des soucis de nullité
		double* Xk = new double[w_size];
		Xk[0] = 1;
		for (int i = 0; i < w_size-1; i++)
		{
			Xk[i + 1] = points[k * 2 + i];
		}
		

		//Calcul de la sortie prédite
		double gXk = 0;
		for (int i = 0; i < w_size; i++)
		{
			gXk += w[i] * Xk[i];
		}

		//Positif ou négatif ?
		gXk = (gXk >= 0) ? 1 : -1;

		//Mise a jour
		for (int i = 0; i < w_size; i++)
		{
			w[i] += 0.01 * (yk - gXk) * Xk[i];
		}

		return w;
	}

}

#pragma endregion


#pragma region PMC

DLLEXPORT void initPMC(int* neuronsTab, int sizeNeuronTab, int maxNumberLayer, float* X, float* deltas, float* W)
{
	// INIT RANDOM GENERATOR BETWEET -1 AND 1
	std::default_random_engine generator;
	std::uniform_real_distribution<float> distribution(-1, 1);

	for (int l = 0; l < sizeNeuronTab; l++)
	{
		if (l == 0)
			continue;
		for (int i = 0; i < neuronsTab[l - 1] + 1; i++)
		{
			for (int j = 0; j < neuronsTab[l] + 1; j++)
			{
				W[l * maxNumberLayer * maxNumberLayer + i * maxNumberLayer + j] = j == 0 ? 0.f : distribution(generator);
			}
		}
	}

	for (int l = 0; l < sizeNeuronTab; l++)
	{
		for (int j = 0; j < neuronsTab[l] + 1; j++)
		{
			deltas[l * maxNumberLayer + j] = 0.f;
			X[l * maxNumberLayer + j] = j == 0 ? 1.0f : 0.f;
		}
	}

}

void propagatePMC(double* inputs, bool is_classification, int* d, int sizeD, double* X, int maxNumberLayer, double* W, int L)
{
	for (int j = 1; j < d[0] + 1; j++)
		X[0 * maxNumberLayer + j] = inputs[j - 1];

	for (int l = 1; l < sizeD; l++)
	{
		for (int j = 1; j < d[l] + 1; j++)
		{
			float total = 0.f;
			for (int i = 0; i < d[l - 1] + 1; i++)
				total += W[l * maxNumberLayer * maxNumberLayer + i * maxNumberLayer + j] * X[(l - 1) * maxNumberLayer + i];

			X[l * maxNumberLayer + j] = total;
			if (is_classification || l < L)
				X[l * maxNumberLayer + j] = tanh(total);
		}
	}
}

DLLEXPORT double* predictPMC(double* inputs, bool is_classification, int* d, int sizeD, int maxNumberLayer, double* X, double *W)
{
	int L = sizeD - 1;
	double* new_arr = new double[d[L]];

	propagatePMC(inputs, is_classification, d, sizeD, X, maxNumberLayer, W, L);
	memcpy(new_arr, &X[L * maxNumberLayer + 1], d[L] * sizeof(double));
	return new_arr;
}

DLLEXPORT void PMCTraining(int sizeT,
	double* X_train,
	int sizeX_train,
	double* Y_train,
	int sizeY_train,
	bool is_classification,
	int* d,
	int sizeD,
	int maxNumberLayer,
	double* X,
	double* deltas,
	double* W,
	float alpha = 0.01,
	int nbIter = 10000)
{
	std::default_random_engine generator;
	std::uniform_int_distribution<int> distribution(0, sizeT);

	int L = sizeD - 1;

	for (int it = 0; it < nbIter; ++it)
	{
		int k = distribution(generator);
		double* Xk = &X_train[k * sizeX_train];
		double* Yk = &Y_train[k * sizeY_train];

		propagatePMC(Xk, is_classification, d, sizeD, X, maxNumberLayer, W, L);
		for (int j = 1; j < d[L] + 1; ++j)
		{
			deltas[L * maxNumberLayer + j] = X[L * maxNumberLayer + j] - Yk[j - 1];
			if (is_classification)
				deltas[L * maxNumberLayer + j] = deltas[L * maxNumberLayer + j] * (1 - (X[L * maxNumberLayer + j] * X[L * maxNumberLayer + j]));
		}

		for (int l = sizeD - 1; l >= 2; --l)
		{
			for (int i = 1; i < d[l - 1] + 1; ++i)
			{
				float total = 0.f;
				for (int j = 1; j < d[l] + 1; ++j)
					total += W[l * maxNumberLayer * maxNumberLayer + i * maxNumberLayer + j] * deltas[l * maxNumberLayer + j];
				deltas[(l - 1) * maxNumberLayer + i] = (1 - (X[(l - 1) * maxNumberLayer + i] * X[(l - 1) * maxNumberLayer + i])) * total;
			}
		}

		for (int l = 1; l < sizeD; ++l)
		{
			for (int i = 0; i < d[l - 1] + 1; ++i)
			{
				for (int j = 1; j <= d[l]; ++j)
					W[l * maxNumberLayer * maxNumberLayer + i * maxNumberLayer + j] += -alpha * X[(l - 1) * maxNumberLayer + i] * deltas[l * maxNumberLayer + j];
			}
		}
	}
}

#pragma endregion
