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
		//Choix d'un index al�atoire dans le tableau d'�l�ments
		int k = rand() % tabLenght;

		//R�cup�ration de sa valeur
		double yk = classification[k];

		//Vecteur d'entr�e avec le biais pour �viter des soucis de nullit�
		double Xk[3] = { 1.0, points[k][0], points[k][1] };

		//Calcul de la sortie pr�dite
		double gXk = 0;
		for (int i = 0; i < 3; i++)
		{
			gXk += w[i] * Xk[i];
		}

		//Positif ou n�gatif ?
		gXk = (gXk >= 0) ? 1 : -1;

		//Mise a jour
		for (int i = 0; i < 3; i++)
		{
			w[i] += 0.01 * (yk - gXk) * Xk[i];
		}

		return w;
	}
}
