#include "pch.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DLLEXPORT extern "C" __declspec(dllexport)


DLLEXPORT double* linearTraining(double** points, double* classification, int tabLenght, int maxIteration, double* w) {

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
		double Xk[3] = { 1.0, points[k][0], points[k][1] };

		//Calcul de la sortie prédite
		double gXk = 0;
		for (int i = 0; i < 3; i++)
		{
			gXk += w[i] * Xk[i];
		}

		//Positif ou négatif ?
		gXk = (gXk >= 0) ? 1 : -1;

		//Mise a jour
		for (int i = 0; i < 3; i++)
		{
			w[i] += 0.01 * (yk - gXk) * Xk[i];
		}

		return w;
	}
}
