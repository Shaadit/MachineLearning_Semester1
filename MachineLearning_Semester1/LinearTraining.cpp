#include "pch.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

#define DLLEXPORT extern "C" __declspec(dllexport)


DLLEXPORT double* linearTraining(double** points, double* classification, int tabLenght, int maxIteration, double* w) {

	//Init du temps pour le random
	srand(time(NULL));
	//double* w_res = new double[3];
	std::cout << "---------- C++ cout -----------\n";
	std::cout << "tab length : " << tabLenght << std::endl;
	std::cout << "maxIteration : " << maxIteration << std::endl;
	std::cout << "classification ptr : " << classification << std::endl;
	std::cout << "w ptr : " << w << std::endl;
	std::cout << "w ptr null ? : " << (w == nullptr) << std::endl;
	std::cout << "--------------------------------\n";
	//Boucle d'entrainement
	for (int iteration = 0; iteration < maxIteration; iteration++)
	{
		//Choix d'un index al�atoire dans le tableau d'�l�ments
		int k = rand() % tabLenght;

		//R�cup�ration de sa valeur
		//double yk = 0;
		//double test = classification->value
		double yk = classification == nullptr ? 0 : 0;

		// acces a *classification ou classification[k] == BUG
		// bug au niveau de la lecture des donnees pointeurs provenant de python 
		// valeur int correctes mais pas les pointeurs 
		return w;

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
