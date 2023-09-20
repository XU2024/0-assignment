{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOAVpvGTyHGZnuEJiNwtuJr",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/XU2024/0-assignment/blob/main/PS_2.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 7,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "-Qrf_sMJUmws",
        "outputId": "f79a98fc-5435-4781-8126-17da78b4f755"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "-1.7418198158321052\n"
          ]
        }
      ],
      "source": [
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import timeit\n",
        "## 2\n",
        "L=100\n",
        "e = 1.602*10**(-19)\n",
        "ep = 8.854*10**(-12)\n",
        "beta = e/(4*np.pi*ep)\n",
        "V_total = 0\n",
        "a = 2.82*10**(-10) ##0.282nm\n",
        "for i in range(-L,L+1):\n",
        "    for j in range(-L, L+1):\n",
        "        for k in range(-L, L+1):\n",
        "            if i==0 and j==0 and k==0:\n",
        "                continue  ## skip i+j=k=0\n",
        "            elif abs(i+j+k) % 2 == 0:\n",
        "                V_total = V_total+ beta/(a*np.sqrt(i**2+j**2+k**2))\n",
        "            else:\n",
        "                V_total = V_total- beta/(a*np.sqrt(i**2+j**2+k**2))\n",
        "M = (V_total/beta)*a\n",
        "print(M)\n",
        "\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "testcode='''\n",
        "import numpy as np\n",
        "L=10\n",
        "e = 1.602*10**(-19)\n",
        "ep = 8.854*10**(-12)\n",
        "beta = e/(4*np.pi*ep)\n",
        "V_total = 0\n",
        "a = 2.82*10**(-10) ##0.282nm\n",
        "for i in range(-L,L+1):\n",
        "    for j in range(-L, L+1):\n",
        "        for k in range(-L, L+1):\n",
        "            if i==0 and j==0 and k==0:\n",
        "                continue  ## skip i+j=k=0\n",
        "            elif abs(i+j+k) % 2 == 0:\n",
        "                V_total = V_total+ beta/(a*np.sqrt(i**2+j**2+k**2))\n",
        "            else:\n",
        "                V_total = V_total- beta/(a*np.sqrt(i**2+j**2+k**2))\n",
        "M = (V_total/beta)*a  '''\n",
        "print(timeit.timeit(stmt=testcode))"
      ],
      "metadata": {
        "id": "a3Jb0xhvkGPg"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "L=10\n",
        "e = 1.602*10**(-19)\n",
        "ep = 8.854*10**(-12)\n",
        "beta = e/(4*np.pi*ep)\n",
        "V_total = 0\n",
        "a = 2.82*10**(-10) ##0.282nm\n",
        "\n",
        "-L <= int(i)<=L\n",
        "-L<=int(j) <=L\n",
        "-L<= int (k) <=L\n",
        "while(i!=0 and j!=0 and k!=0):\n",
        "  if abs(i+j+k) % 2 == 0:\n",
        "    V_total = V_total+ beta/(a*np.sqrt(i**2+j**2+k**2))\n",
        "  else:\n",
        "    V_total = V_total- beta/(a*np.sqrt(i**2+j**2+k**2))\n",
        "M = (V_total/beta)*a\n",
        "print(M)"
      ],
      "metadata": {
        "id": "jaOGerNCU09F"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "##3\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import cmath\n",
        "\n",
        "graph = np.zeros((100, 100)) ## the matrix of 100*100\n",
        "def Mandelbrost(C):\n",
        "  z=0\n",
        "  N=0\n",
        "  while abs(z)<=2 and N<=100:\n",
        "    z = z**2+C\n",
        "    N += 1\n",
        "  return N    ## the number of iteration when it stops\n",
        "\n",
        "\n",
        "x = np.linspace(-2, 2, 100) ## width = 100\n",
        "y = np.linspace(-2, 2, 100) ## height = 100\n",
        "for i in range(100):\n",
        "  for j in range(100):\n",
        "    c = complex(x[i], y[j])\n",
        "    graph[i, j] = Mandelbrost(c)\n",
        "\n",
        "plt.imshow(graph.T, extent=(-2, 2, -2, 2)) ## the size\n",
        "plt.colorbar()\n",
        "plt.title('Mandelbrot test')\n",
        "plt.xlabel('real')\n",
        "plt.ylabel('imaginary')\n",
        "plt.show()\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 472
        },
        "id": "bYkF-tcyU8yN",
        "outputId": "833197e1-4c05-4c04-b658-a0e6f40bac6c"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 2 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAhMAAAHHCAYAAAAF5NqAAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAABbp0lEQVR4nO3deVxU5f4H8M+AMKBsoqyFiktiai6khplikrhkkl7N5Ve4pGVYKt660k3NyriWpWmWWoZ2b2Z5c7lZaYZLV8UNtTSV1FDcwC1WFZB5fn94GeYMMDBzziyH+bxfr/NqzjnPOec7EzgPz/N9nkcjhBAgIiIispCLvQMgIiIidWNlgoiIiGRhZYKIiIhkYWWCiIiIZGFlgoiIiGRhZYKIiIhkYWWCiIiIZGFlgoiIiGRhZYKIiIhkYWWCSAVef/11aDQai64dM2YMmjVrpt8/e/YsNBoN5s+fr1B0ROTsWJkgp7dy5UpoNBpoNBrs2rWr0nkhBMLCwqDRaPD444/bIUJ1e/vtt7Fhw4Zalb106RJef/11HDlyxKoxrV69GgsXLrTqM4icCSsTRP/j4eGB1atXVzq+c+dOXLhwAVqt1g5RqZ+5lYk5c+awMkGkMqxMEP3PgAEDsHbtWty5c0dyfPXq1YiMjERwcLCdInMsRUVF9g6BiBwMKxNE/zNy5Ehcv34dW7du1R8rKSnBv//9b4waNarKa+bPn4/u3bujUaNG8PT0RGRkJP79739XKqfRaDB58mRs2LAB7dq1g1arRdu2bbF58+ZKZXft2oUuXbrAw8MDLVq0wLJly6qN+V//+hciIyPh6ekJf39/jBgxAufPn6/1e16wYAGaNm0KT09P9OrVC8eOHZOcHzNmDLy8vHDmzBkMGDAA3t7eGD16NIC7lYrp06cjLCwMWq0WrVu3xvz582G4ELFGo0FRURFWrVql70oaM2ZMlbHs2LEDXbp0AQCMHTtWX37lypX6Mvv27UO/fv3g6+uL+vXro1evXti9e7fkPgUFBZg6dSqaNWsGrVaLwMBAPPbYYzh06BAAIDo6Gt999x3OnTunf4ZhTgkRma+evQMgchTNmjVDVFQUvvzyS/Tv3x8A8MMPPyAvLw8jRozAokWLKl3zwQcf4IknnsDo0aNRUlKCNWvWYNiwYdi0aRMGDhwoKbtr1y6sW7cOL7zwAry9vbFo0SIMHToUWVlZaNSoEQDg6NGj6Nu3LwICAvD666/jzp07mD17NoKCgio9e+7cuZg5cyaGDx+OZ599FlevXsXixYvRs2dPHD58GH5+fibf7+eff46CggIkJCTg9u3b+OCDD/Doo4/i6NGjkufduXMHsbGx6NGjB+bPn4/69etDCIEnnngC27dvx/jx49GxY0ds2bIFL7/8Mi5evIgFCxYAAP75z3/i2WefRdeuXTFx4kQAQIsWLaqMp02bNnjjjTcwa9YsTJw4EY888ggAoHv37gCAbdu2oX///oiMjMTs2bPh4uKClJQUPProo/jvf/+Lrl27AgCef/55/Pvf/8bkyZNx//334/r169i1axdOnDiBzp074+9//zvy8vJw4cIFfZxeXl4mPysiqoEgcnIpKSkCgDhw4ID48MMPhbe3t7h586YQQohhw4aJ3r17CyGEaNq0qRg4cKDk2vJy5UpKSkS7du3Eo48+KjkOQLi7u4vTp0/rj/3yyy8CgFi8eLH+WFxcnPDw8BDnzp3THzt+/LhwdXUVhr+uZ8+eFa6urmLu3LmS5xw9elTUq1dPcjw+Pl40bdpUv5+ZmSkACE9PT3HhwgX98X379gkAYtq0aZJrAYgZM2ZInrNhwwYBQLz11luS43/5y1+ERqORvM8GDRqI+Ph4URsHDhwQAERKSorkuE6nE61atRKxsbFCp9Ppj9+8eVOEh4eLxx57TH/M19dXJCQkmHzOwIEDJZ8JEcnDbg4iA8OHD8etW7ewadMmFBQUYNOmTdV2cQCAp6en/vWff/6JvLw8PPLII/omdUMxMTGSv8ofeOAB+Pj44I8//gAAlJWVYcuWLYiLi0OTJk305dq0aYPY2FjJvdatWwedTofhw4fj2rVr+i04OBitWrXC9u3ba3yvcXFxuOeee/T7Xbt2Rbdu3fD9999XKjtp0iTJ/vfffw9XV1e89NJLkuPTp0+HEAI//PBDjc83x5EjR3Dq1CmMGjUK169f17/foqIi9OnTBz///DN0Oh0AwM/PD/v27cOlS5cUjYGIqsduDiIDAQEBiImJwerVq3Hz5k2UlZXhL3/5S7XlN23ahLfeegtHjhxBcXGx/nhVc0IYVhDKNWzYEH/++ScA4OrVq7h16xZatWpVqVzr1q0lX/KnTp2CEKLKsgDg5uZW/Zv8n6quve+++/D1119LjtWrVw/33nuv5Ni5c+cQGhoKb29vyfE2bdrozyvp1KlTAID4+Phqy+Tl5aFhw4Z45513EB8fj7CwMERGRmLAgAF45pln0Lx5c0VjIqIKrEwQGRk1ahQmTJiA7Oxs9O/fv9rcg//+97944okn0LNnT3z00UcICQmBm5sbUlJSqhxi6urqWuV9hEHCYm3pdDpoNBr88MMPVd5XyRwArVYLFxf7NmKWtzq8++676NixY5Vlyt/z8OHD8cgjj2D9+vX48ccf8e6772LevHlYt26dPheGiJTFygSRkSeffBLPPfcc9u7di6+++qract988w08PDywZcsWyRwUKSkpFj03ICAAnp6e+r/CDWVkZEj2W7RoASEEwsPDcd9991n0vKqe8/vvv9dqZEPTpk3x008/oaCgQNI6cfLkSf35cubM3Fld2fLuIR8fH8TExNR4n5CQELzwwgt44YUXcOXKFXTu3Blz587VVyYsnU2UiKrGnAkiI15eXvj444/x+uuvY9CgQdWWc3V1hUajQVlZmf7Y2bNnaz1BU1X3i42NxYYNG5CVlaU/fuLECWzZskVSdsiQIXB1dcWcOXMqtWwIIXD9+vUan7dhwwZcvHhRv79//37s27evVn+9DxgwAGVlZfjwww8lxxcsWACNRiO5R4MGDZCbm1vjPcvLAqhUPjIyEi1atMD8+fNRWFhY6bqrV68CuJt3kpeXJzkXGBiI0NBQSTdUgwYNKpUjIsuxZYKoCqb65ssNHDgQ77//Pvr164dRo0bhypUrWLJkCVq2bIlff/3VoufOmTMHmzdvxiOPPIIXXngBd+7cweLFi9G2bVvJPVu0aIG33noLSUlJOHv2LOLi4uDt7Y3MzEysX78eEydOxF//+leTz2rZsiV69OiBSZMmobi4GAsXLkSjRo3wyiuv1BjnoEGD0Lt3b/z973/H2bNn0aFDB/z444/YuHEjpk6dKkk0jYyMxE8//YT3338foaGhCA8PR7du3aq8b4sWLeDn54elS5fC29sbDRo0QLdu3RAeHo5PP/0U/fv3R9u2bTF27Fjcc889uHjxIrZv3w4fHx98++23KCgowL333ou//OUv6NChA7y8vPDTTz/hwIEDeO+99yQxffXVV0hMTESXLl3g5eVlsuJIRDWw51ASIkdgODTUlKqGhq5YsUK0atVKaLVaERERIVJSUsTs2bOF8a8WgCqHKzZt2rTSsMmdO3eKyMhI4e7uLpo3by6WLl1a5T2FEOKbb74RPXr0EA0aNBANGjQQERERIiEhQWRkZOjLVDc09N133xXvvfeeCAsLE1qtVjzyyCPil19+kdw/Pj5eNGjQoMrPo6CgQEybNk2EhoYKNzc30apVK/Huu+9Khm4KIcTJkydFz549haenpwBQ4zDRjRs3ivvvv1/Uq1ev0jDRw4cPiyFDhohGjRoJrVYrmjZtKoYPHy5SU1OFEEIUFxeLl19+WXTo0EF4e3uLBg0aiA4dOoiPPvpI8ozCwkIxatQo4efnJwBwmCiRTBohLMj+IiIiIvof5kwQERGRLKxMEBERkSysTBAREZEsqqlMJCcno0uXLvD29kZgYCDi4uIqjb2vytq1axEREQEPDw+0b9++yqmCiYiIyHKqqUzs3LkTCQkJ2Lt3L7Zu3YrS0lL07dsXRUVF1V6zZ88ejBw5EuPHj8fhw4cRFxeHuLi4SsssExERkeVUO5rj6tWrCAwMxM6dO9GzZ88qyzz11FMoKirCpk2b9MceeughdOzYEUuXLrVVqERERHWaaietKp+9zt/fv9oyaWlpSExMlBwrn2GwOsXFxZKZ8nQ6HW7cuIFGjRpxCl4iIhUSQqCgoAChoaFWW2fm9u3bKCkpUeRe7u7u8PDwUOReNmPXWS4sVFZWJgYOHCgefvhhk+Xc3NzE6tWrJceWLFkiAgMDq72mfHIgbty4ceNWt7bz588r8h1k7NatWyI40FWxOIODg8WtW7dq9eydO3eKxx9/XISEhAgAYv369ZLzOp1OzJw5UwQHBwsPDw/Rp08f8fvvv0vKXL9+XYwaNUp4e3sLX19fMW7cOFFQUGDWZ6DKlomEhAQcO3YMu3btUvzeSUlJktaMvLw8NGnSBD0wAPVQ87LORETkWO6gFLvwvWRROiWVlJQg+0oZzqU3g4+3vJaP/AIdmkaeRUlJSa1aJ4qKitChQweMGzcOQ4YMqXT+nXfewaJFi7Bq1SqEh4dj5syZiI2NxfHjx/X3Hz16NC5fvqzPRxw7diwmTpxY5erH1VFdZWLy5MnYtGkTfv75Z9x7770mywYHByMnJ0dyLCcnB8HBwdVeo9VqJStAlqsHN9TTsDJBRKQ64u5/rN1V7eWtgZe3vGfoYN71/fv3r3ZxPiEEFi5ciNdeew2DBw8GAHz++ecICgrChg0bMGLECJw4cQKbN2/GgQMH8OCDDwIAFi9ejAEDBmD+/PkIDQ2tVRyqGc0hhMDkyZOxfv16bNu2DeHh4TVeExUVhdTUVMmxrVu3IioqylphEhGRkyoTOkU2AMjPz5dshrl8tZWZmYns7GzExMToj/n6+qJbt25IS0sDcDe30M/PT1+RAICYmBi4uLhg3759tX6WalomEhISsHr1amzcuBHe3t7Izs4GcPeD8fT0BAA888wzuOeee5CcnAwAmDJlCnr16oX33nsPAwcOxJo1a3Dw4EEsX77cbu+DCC6u9o5AHXRlNZchciA6COjKm0Fk3AMAwsLCJMdnz56N119/3ax7lX9PBgUFSY4HBQXpz2VnZyMwMFByvl69evD399eXqQ3VVCY+/vhjAEB0dLTkeEpKCsaMGQMAyMrKkmTqdu/eHatXr8Zrr72GV199Fa1atcKGDRvQrl07W4VNRERktvPnz8PHx0e/X1X3uyNRTWVC1GI6jB07dlQ6NmzYMAwbNswKEREREVXQQQedAvcAAB8fH0llwhLl+YE5OTkICQnRH8/JyUHHjh31Za5cuSK57s6dO7hx44bJ/EJjqsmZIHIoLq6Wb1Q7/IxJZcqEUGRTSnh4OIKDgyW5g/n5+di3b58+dzAqKgq5ublIT0/Xl9m2bRt0Oh26detW62eppmWCiIiIpAoLC3H69Gn9fmZmJo4cOQJ/f380adIEU6dOxVtvvYVWrVrph4aGhoYiLi4OANCmTRv069cPEyZMwNKlS1FaWorJkydjxIgRtR7JAbAyQUREpAglEzBr6+DBg+jdu7d+v3yepPj4eKxcuRKvvPIKioqKMHHiROTm5qJHjx7YvHmzZA6LL774ApMnT0afPn3g4uKCoUOHYtGiRWbFodq1OWwlPz8fvr6+iMZgzjNBFdiU7tg4EoQM3BGl2IGNyMvLk52HUJXy74nMkyHwljlpVUGBDuERl60Wq7WwZYKoHCsIdYc5/y9Z8SCSjZUJIiIiBdijm8NRsDJBRESkACVGYyg5msOWODSUiIiIZGHLBDkX5kWQMVM/E8ynIDPo/rfJvYcasTJBRESkgDIIlMnMeZB7vb2wMkFERKSAMnF3k3sPNWJlgtSPXRdkLRxiSlQrrEwQEREpgDkTREREJIsOGpRBI/seasShoURERCQLWybI8TEngtSgpp9T5lTUeTpxd5N7DzViZYKIiEgBZQp0c8i93l7YzUFERESysGWCiIhIAc7cMsHKBDke5khQXWT4c838iTpJJzTQCZmjOWReby/s5iAiIiJZ2DJBRESkAHZzEBERkSxlcEGZzAZ/tXaAsTJB9sG8CHJmnJOiThIK5EwI5kwQERGRM2LLBBERkQKYM0FkbezWIKo9498XdnuoQplwQZmQmTOh0um02c1BREREsrBlgoiISAE6aKCT+Te6DupsmmBlgoiISAHMmSBSAvMiiKzD1O8W8ynIAbAyQUREpABlEjDZzUFEROS07uZMyFzoS6XdHBzNQURERLKwZYIsxxwJIvvjnBQOQ6fA2hxqHc2hqpaJn3/+GYMGDUJoaCg0Gg02bNhgsvyOHTug0WgqbdnZ2bYJmIiInEZ5zoTcTY1U1TJRVFSEDh06YNy4cRgyZEitr8vIyICPj49+PzAw0BrhERGRE9PBhfNMqEH//v3Rv39/s68LDAyEn5+f8gE5I3ZtEDk2w99RdnmQjaizPcVMHTt2REhICB577DHs3r3bZNni4mLk5+dLNiIiopqUCY0imxrV6cpESEgIli5dim+++QbffPMNwsLCEB0djUOHDlV7TXJyMnx9ffVbWFiYDSMmIiK1KvtfAqbcTY1U1c1hrtatW6N169b6/e7du+PMmTNYsGAB/vnPf1Z5TVJSEhITE/X7+fn5rFAQERGZUKcrE1Xp2rUrdu3aVe15rVYLrVZrw4gcHHMkiNSLw0ZtSidcoJM5GkPHGTDV4ciRIwgJCbF3GEREVMco0U1RxtEc1ldYWIjTp0/r9zMzM3HkyBH4+/ujSZMmSEpKwsWLF/H5558DABYuXIjw8HC0bdsWt2/fxqeffopt27bhxx9/tNdbICIiqnNUVZk4ePAgevfurd8vz22Ij4/HypUrcfnyZWRlZenPl5SUYPr06bh48SLq16+PBx54AD/99JPkHkRERErQAbJHY+iUCcXmNEKotIPGRvLz8+Hr64toDEY9jZu9w7E+5kgQOQ8nyaG4I0qxAxuRl5cnmcBQKeXfEx8f6gJPL3l/o98qvINJnQ9YLVZrUecYFCIiInIYqurmICIiclRKrK3BtTlInditQeS8OPW2onTQQAe5ORPqnAGTlQkiIiIFOHPLhDqjJiIiIofBlgkiIiIFKDNplTr/xmdlwhkxT4KIjHHqbdl0QgOd3HkmuGooEREROSO2TBARESlAp0A3h06lf+OzMkFERKQAZVYNZWWCHBVzJGpF42b/XwdResfeIRDdxRwKMoP9//UkIiKqA8qgQZnMSafkXm8vrEwQEREpwJm7OdQZNRERETkMtkzURXU8R8IRchusxZrvjfkYJAtzKGpUBvndFGr9VOvuv8pEREQ25MzdHKxMEBERKcCZF/piZaKuUHnXRl3uunAU5nzG7BKhGnH5cjLAf8GJiIgUIKCBTmbOhODQUCIiIuflzN0c6oyaiIiIUFZWhpkzZyI8PByenp5o0aIF3nzzTQgh9GWEEJg1axZCQkLg6emJmJgYnDp1StE42DKhVirIkWAehHoxv4LMwmGjAOyzBPm8efPw8ccfY9WqVWjbti0OHjyIsWPHwtfXFy+99BIA4J133sGiRYuwatUqhIeHY+bMmYiNjcXx48fh4eEhK95y/NeeiIhIAWUKrBpq7vV79uzB4MGDMXDgQABAs2bN8OWXX2L//v0A7rZKLFy4EK+99hoGDx4MAPj8888RFBSEDRs2YMSIEbLiLcduDiIiIpXq3r07UlNT8fvvvwMAfvnlF+zatQv9+/cHAGRmZiI7OxsxMTH6a3x9fdGtWzekpaUpFgdbJoiIiBSgZDdHfn6+5LhWq4VWq61UfsaMGcjPz0dERARcXV1RVlaGuXPnYvTo0QCA7OxsAEBQUJDkuqCgIP05JbAyQRZjToQyNK6Ol/8iymrf523q54D5FORMdHCBTmaDf/n1YWFhkuOzZ8/G66+/Xqn8119/jS+++AKrV69G27ZtceTIEUydOhWhoaGIj4+XFYs5+G1ARETkYM6fPw8fHx/9flWtEgDw8ssvY8aMGfrch/bt2+PcuXNITk5GfHw8goODAQA5OTkICQnRX5eTk4OOHTsqFi9zJoiIiBRQJjSKbADg4+Mj2aqrTNy8eRMuLtKvcldXV+h0OgBAeHg4goODkZqaqj+fn5+Pffv2ISoqSrH3zpYJtbDTUFB2ZVhGqa4LTT3p56+p76l/Lcp00sJ3jLoUTHRVCOOyxs81Eb9SXSAAu0HqLCcdKmqPoaGDBg3C3Llz0aRJE7Rt2xaHDx/G+++/j3HjxgEANBoNpk6dirfeegutWrXSDw0NDQ1FXFycrFgN8ZuCiIhIAUKBVUOFmdcvXrwYM2fOxAsvvIArV64gNDQUzz33HGbNmqUv88orr6CoqAgTJ05Ebm4uevTogc2bNys2xwQAaIThNFlUSX5+Pnx9fRGNwaincbNfIGyZUBW1t0yYYk7LRI33YsuEc7Bzy8QdUYod2Ii8vDxJHoJSyr8nJu4cBncved8TJYWlWN5rrdVitRZ+UxARESmgDBqUyVyoS+719sLKhCOzQ2sEWyJqx1rDOY1bIoyJ4pKKsr7e0pO3i6stC0DSUmH8HHNaKpTKpwA4rNRpOMly5Tphfs5DVfdQI47mICIiIln4ZygREZECdAokYMq93l5UFfXPP/+MQYMGITQ0FBqNBhs2bKjxmh07dqBz587QarVo2bIlVq5cafU4iYjI+eigUWRTI1W1TBQVFaFDhw4YN24chgwZUmP5zMxMDBw4EM8//zy++OILpKam4tlnn0VISAhiY2NtELFjYl5E7dhqmuua8iSqZTRRjfDxkt43v1B63mD0h0brLr1XfoG0rIWjPYw/MzkjP4x/TplDQeS4VPWt0r9/f/1KaLWxdOlShIeH47333gMAtGnTBrt27cKCBQucujJBRETKM5zBUs491EhV3RzmSktLkyy7CgCxsbGKLrtKREQEVORMyN3USFUtE+bKzs6uctnV/Px83Lp1C56enpWuKS4uRnFxxRA742VgrcpKQ0HZrVE7DtmtYRyT4bU3b0lOiSB/6XNKpHP5F7UP1r/2+u2KyecY/m0kZ4Ira3V7sMtDxZx0qu26Tp1VICtKTk6Gr6+vfjNeBpaIiKgqOmj063NYvKk0AbNOVyaCg4ORk5MjOZaTkwMfH58qWyUAICkpCXl5efrt/PnztgiViIhUTigwkkOotDJRp9u/o6Ki8P3330uObd261eSyq1qtttqlXomIiKpjj1VDHYWqKhOFhYU4ffq0fj8zMxNHjhyBv78/mjRpgqSkJFy8eBGff/45AOD555/Hhx9+iFdeeQXjxo3Dtm3b8PXXX+O7776z11uQYo6EzdkqL8IeXAqkORQ3I6T5Ql5Hs/WvRZG0rClypt6udC+Fcii4tHkdwhyKOkFV3zoHDx5E79699fuJiYkAgPj4eKxcuRKXL19GVlaW/nx4eDi+++47TJs2DR988AHuvfdefPrppxwWSkREinPmGTBVVZmIjo6GqRXTq5rdMjo6GocPH7ZiVERERM7dzaHOKhARERE5DFW1TFAF5kXUjiPkSCg2rwQAjatB/d9Dmiis8656hFI54e5WcR/j5cpN/VNglNdgrRwKOXNQVLov56QgO1BibQ21Dg3lNxIREZEC2M1BREREZCG2TBARESnAmVsmWJmwJRnzSjBHonYcIUcCkJEnYZwj0UCaB2GY9wBXacPi6VF+kv1TT38s2e/ff2TFfY3W7TD+50vkFaC2DN+rI8xBUem+XMpcXQz/nVTZnBPOXJlgNwcRERHJwj93iYiIFODMLROsTJDqOULXhlLDPzVad8kp4Sntjrh9j4/+9dk46TP/GCLt1jCWE+Wnfx2UJj2nKTFq+jc13baJ7oeaPgdzukHY7UFqIyB/aGf10zI6NlYmiIiIFODMLRPMmSAiIiJZ2DJBRESkAGdumWBlwoFxOGjV7JEjUWNOhFIxGecUlOkku6XeFXH8MWSZWbc+NKv6nIqBPeIk+yb7bY3fqxm5DEpOxa0U5lCQUpy5MsFuDiIiIpKFf/oSEREpwJlbJliZICIiUoAQGgiZlQG519sLKxPWZsYU2syRuMsR5o0AjPr35cRUQ76FZFlx45wCT+m8E/9dYl6ehKUMY6q0PLlxnoOpz6aGfApzpuK21nLl5MCM//1U2fTazoTfXkRERArQQSN70iq519sLKxNEREQKYM4E2QW7NSo4/HBP49U8XU0PhBJGQzpNXmt473rSz0Fzq0Sy3z3xef3rPe8vNRmDKRPOP2y6gOEU35B2P9TY7WGopv+vhs8xnj67uNj0tVZi+HvJYaJEtcNvMyIiIgUwAZOIiIhkYTcHERERycKWCVKOiaGgzJGoYK0cCbOWAjdmFJOmoa/+ta6+h+Tc70mekv3WM/+UXltSWrFjlAdx7ZF7JPuNDl7TvzZcYhwAtFdvSva9zplYGrwG3f42Sf+64clCyTlN/VJpYe+K9+dSIH1maVhDyb57xiX9a1O5IlUxXHJdNPaTnruYI9k3HDpqreXJjXGqbQfDoaIOi99uREREChAKdHOwZYKIiMiJCQDC5Cp5tbuHGnGhLyIiIpKFLRNkE0rmSMjKizBkHJPxfQ3yHvK6BktONUyVNkWWBvlK9i/1bKB/3WS9tO9/3uvSKbHHbn1W/9r9ujSmjLErqgjcMvvmVb8EuSmFutuS/a4fJ0r2w89X5FdIckUAwN1Nuq+T5lQIg/OaW9J5JYRxDovhOQdYupzImA4aaDgDJhEREVnKmUdzsJuDiIiIZGHLhJU503BQOV0ZinVd1MTCGP1+uS7ZvxrVWLL/4zerJPvt339B//pmS3/JuWhPaVN/5hPLLYrJVrxcpMNijyd8JNlvHvKc/nXEkhuSc9+nrpXs9x43QbIvDP6c8cwqkJzTFBkNgzUx/JNDRckR6IQGGk5aRURERJYSQoHRHCodzsFuDiIiIpKFLRNEREQKcOYETFYm5DIxfXZdZ06OhM1yIoyZE2MNy4obutHedFvk0cSPTJ6vS/4YUjHUtc+68SbLbv/sE8l+36Hx+tcuudKcCVPDP41/nuw1VJTLlduZg02v7cyVCdV1cyxZsgTNmjWDh4cHunXrhv3791dbduXKldBoNJLNw8Oj2vJERESWKl81VO6mRqqqTHz11VdITEzE7NmzcejQIXTo0AGxsbG4cuVKtdf4+Pjg8uXL+u3cuXM2jJiIiKjuU1Vl4v3338eECRMwduxY3H///Vi6dCnq16+Pzz77rNprNBoNgoOD9VtQUJANIyYiImdRPppD7qZGqsmZKCkpQXp6OpKSkvTHXFxcEBMTg7S0tGqvKywsRNOmTaHT6dC5c2e8/fbbaNu2rdXirGvzSqgiL8KQOfNIGMdrvG88FbSBR7sfNSMo55H6L/Om/y7zrPjM3RTMezD8ubXWnBNExu5WBuTmTCgUjI2ppmXi2rVrKCsrq9SyEBQUhOzs7Cqvad26NT777DNs3LgR//rXv6DT6dC9e3dcuHCh2ucUFxcjPz9fshEREVH1VFOZsERUVBSeeeYZdOzYEb169cK6desQEBCAZcuWVXtNcnIyfH199VtYWJgNIyYiIrUqH80hd1Mj1VQmGjduDFdXV+TkSFdgzMnJQXBwcDVXSbm5uaFTp044ffp0tWWSkpKQl5en386fPy8rbiIicg5CoU2NHKCTu3bc3d0RGRmJ1NRUxMXFAQB0Oh1SU1MxefLkWt2jrKwMR48exYABA6oto9VqodVqlQhZFcxdT0OSF1HTtdbqq5aznLlB/CJUur7GtU5+kv3Ceyr+QghJky6P/UmYdM0Jqp3Hho+R7Htcul51QaByDosh5kEQORTVVCYAIDExEfHx8XjwwQfRtWtXLFy4EEVFRRg7diwA4JlnnsE999yD5ORkAMAbb7yBhx56CC1btkRubi7effddnDt3Ds8++6w93wYREdVBzjxplaoqE0899RSuXr2KWbNmITs7Gx07dsTmzZv1SZlZWVlwcanoufnzzz8xYcIEZGdno2HDhoiMjMSePXtw//332+stEBFRXaVEP4VK+zk0Qqh1IIpt5Ofnw9fXF9EYjHqaykMFNQ7YJaLoUuBG99I0qK9/LYpuWvwcRZkxJFUyZbaH9P9dyb3SpcK3fr1STlRUC11em6R/HfhjlvSk0VBRUVxSsWPUzWFqOm17DQ3l9Np2UM102ndEKXZgI/Ly8uDj46P4Y8u/J5qv/Dtc6subZVl38zb+GDPXarFai2oSMImIiMgxqaqbg4iIyFEpMYOlWvsK2DJBRESkAHvNM3Hx4kX83//9Hxo1agRPT0+0b98eBw8eNIhLYNasWQgJCYGnpydiYmJw6tQpJd86WyYsYodlx+XkQZi8r4wpsDVad8m+KNPV/lqj5b5LwyvmCnHLlM5oas59a3qO5L5G02W73iq1+DlkmQNvfax/3e/o05JzrteNliQPqshp0WRerPUzjH93OL12HWb4b7OdlyO3lT///BMPP/wwevfujR9++AEBAQE4deoUGjZsqC/zzjvvYNGiRVi1ahXCw8Mxc+ZMxMbG4vjx44qtpM3KBBERkRKE5u4m9x5mmDdvHsLCwpCSkqI/Fh4eXnE7IbBw4UK89tprGDx4MADg888/R1BQEDZs2IARI0bIi/d/2M1BRESkAHusGvqf//wHDz74IIYNG4bAwEB06tQJn3zyif58ZmYmsrOzERMToz/m6+uLbt26mVwk01ysTBARETkY4wUni4uLqyz3xx9/4OOPP0arVq2wZcsWTJo0CS+99BJWrVoFAPqFMM1ZJNMS7OZwIA6RF1FTDIZ5BkY5B5oSac6Bzs9b/9olV9r/jXrS57gWGPyiGM3/oLlj1PdpzlLVlZYZr3juib9K55XIfPwTkP1s3vhPyf6DsydJ9hsfqljBVw1zBGrcpD97nHfCCSg4aZXxIpOzZ8/G66+/Xqm4TqfDgw8+iLfffhsA0KlTJxw7dgxLly5FfHy8zGBqz6KWidmzZ+PcuXNKx0JERKRaSo7mOH/+vGTRyaSkpCqfGRISUmlW5zZt2iAr6+4kcOULYcpZJLM2LKpMbNy4ES1atECfPn2wevXqaptfiIiIyHw+Pj6SrboFKB9++GFkZGRIjv3+++9o2rQpgLvJmMHBwUhNTdWfz8/Px759+xAVFaVYvBZ1cxw5cgSHDx9GSkoKpkyZgoSEBIwYMQLjxo1Dly5dFAvOURk3X9b6Oit1Y1R6jozhnpUY3+vmLf1L4eMlOVXQQdon51pcMaTTQ2vUrfFnkfS+rhUN19ceuUdyqtHBa9KyRkNFNTdvV467nFF3CgzWbmm2zqjs49XfhqzvsZFjJftB56TDP0XRLRA5PBtPOjVt2jR0794db7/9NoYPH479+/dj+fLlWL58OQBAo9Fg6tSpeOutt9CqVSv90NDQ0FD9CtxKsDgBs1OnTli0aBEuXbqEFStW4MKFC3j44YfxwAMP4IMPPkBeXp5iQRIRETk6e0xa1aVLF6xfvx5ffvkl2rVrhzfffBMLFy7E6NGj9WVeeeUVvPjii5g4cSK6dOmCwsJCbN68WbE5JgAFRnMIIVBaWoqSkhIIIdCwYUN8+OGHCAsLw1dffaVEjERERI5PKLSZ6fHHH8fRo0dx+/ZtnDhxAhMmTJCc12g0eOONN5CdnY3bt2/jp59+wn333WfZe6yGxZWJ9PR0TJ48GSEhIZg2bRo6deqEEydOYOfOnTh16hTmzp2Ll156SclYiYiIyAFZ1Lnevn17nDx5En379sWKFSswaNAguBrlA4wcORJTpkxRJEg1U2WeRG2faTQUtEFWoWR/87df6F+Hb5LWlEO2eUv2XYsrquP75n0sORf9rPTa/OfzJfsNF1ZkJHucvW4yZqGtGM56tUPlJeXJfq619ZTsh3LAGKmOBvIHLqth4HNlFn0DDR8+HOPGjcM999xTbZnGjRtDp7N8TQUiIiJVUXCeCbUxu5ujtLQUK1euRH5+fs2FiYiIqM4zu2XCzc0Nt2+bGIpHRETkjJy4ZcKibo6EhATMmzcPn376KerZoa/ekakuR8I43prua3jepfYNW5WmqjZjTocdn9YwzfW/Kl72GyxdxtqlUFrxFe4V8R+b8lHtgyCrO/ya9P/HY79J551wO37BluEQmc8Oq4Y6Cou+kQ4cOIDU1FT8+OOPaN++PRo0aCA5v26d8WxAREREVFdZVJnw8/PD0KFDlY6FiIhItSxZQryqe6iRRZWJlJQUpeNQLdV1awDSrg2j+2pcjboujM7/2aNiJTuvc9Lpjf8YKh3aZw/GK08ad3uUervbMhySwf3MFcm+MGe1WBOMf2dFWVk1JYnM5MQ5E7JnwCQiIiLnZvGfu//+97/x9ddfIysrCyUlJZJzhw4dkh0YERGRqjhxAqZFLROLFi3C2LFjERQUhMOHD6Nr165o1KgR/vjjD/Tv31/pGImIiByeRiizqZFFLRMfffQRli9fjpEjR2LlypV45ZVX0Lx5c8yaNQs3btxQOkaH45B5EmbEpNEa5A3UN8pzMJoi23gJb79DV/Wvi1o3kpw7PWpprWOwlcw46TLpIbuV6XcnG1AoR8JRaNwqfp9Fad16b/Q/zJkwT1ZWFrp37w4A8PT0REFBAQDg6aefxpdffqlcdEREROTwLKpMBAcH61sgmjRpgr179wIAMjMzIdQ6roWIiEiO8pwJuZsKWVSZePTRR/Gf//wHADB27FhMmzYNjz32GJ566ik8+eSTigZIRESkCkKhTYUsyplYvny5fkXQhIQENGrUCHv27METTzyB5557TtEAnUmlHAlLczPMmBJbV99DGoNWuiy34fTTxnYuW25+bDaWMVa6nDnGVl2O7K/XcxMl+w3KMqUFOB8EkcOyqDLh4uICF4N1GUaMGIERI0YoFhQREZHqOHECpsXzTOTm5mL//v24cuWKvpWi3DPPPCM7MCIiIlVhZcI83377LUaPHo3CwkL4+PhAo6lIGNFoNKxMEBERORGLKhPTp0/HuHHj8Pbbb6N+/fpKx+TwlJr/QTLfAwBRJm3hMTnO3owYjJ8jOWfUD/38f76T7D/R4KZkv8eLzIkh5RiuneJ1/pzknFJrcRDZjBPPgGlRZeLixYt46aWXnLIiQUREVBUlZrBU6wyYFg0NjY2NxcGDB5WOhYiIiFTIopaJgQMH4uWXX8bx48fRvn17uLlJhxM+8cQTigRXlSVLluDdd99FdnY2OnTogMWLF6Nr167Vll+7di1mzpyJs2fPolWrVpg3bx4GDBigXEBarXTfqGnWVBdDSetQyb7xksswXA7ceNprnXGXiEF3hdEU2DovaQuSYdfGhf4BknPG3RrGdi1eZvI8kSmPjZSOzXW/XPEzX6mbj0htmIBpngkTJgAA3njjjUrnNBoNyqw0Hvyrr75CYmIili5dim7dumHhwoWIjY1FRkYGAgMDK5Xfs2cPRo4cieTkZDz++ONYvXo14uLicOjQIbRr184qMRIRETkbi7o5dDpdtZu1KhIA8P7772PChAkYO3Ys7r//fixduhT169fHZ599VmX5Dz74AP369cPLL7+MNm3a4M0330Tnzp3x4YcfWi1GIiJyThoosGqovd+EhSyqTNhDSUkJ0tPTERMToz/m4uKCmJgYpKWlVXlNWlqapDxwN9+juvIAUFxcjPz8fMlGRERE1at1N8eiRYswceJEeHh4YNGiRSbLvvTSS7IDM3bt2jWUlZUhKChIcjwoKAgnT56s8prs7Owqy2dnZ1f7nOTkZMyZM0d+wERE5Fw4NLRmCxYswOjRo+Hh4YEFCxZUW06j0VilMmErSUlJSExM1O/n5+cjLCzMjhEREZEqMAGzZpmZmVW+tpXGjRvD1dUVOTk5kuM5OTkIDg6u8prg4GCzygOAVquF1niEBhEREVVLNTkT7u7uiIyMRGpqqv6YTqdDamoqoqKiqrwmKipKUh4Atm7dWm15IiIii3EJcvMYdgMY0mg08PDwQMuWLTF48GD4+/vLCq6q58bHx+PBBx9E165dsXDhQhQVFWHs2Ltj15955hncc889SE5OBgBMmTIFvXr1wnvvvYeBAwdizZo1OHjwIJYvV3Dp7OJik6dFcUm159wzLknLmppO2/g+pqbTNgrJ5U71I2zu/eGqZP8/k6RzUpiaTptzTpC5tn6ZItk3nE7b9bx0nhXB2bRJZZx5BkyLKhOHDx/GoUOHUFZWhtatWwMAfv/9d7i6uiIiIgIfffQRpk+fjl27duH+++9XLNinnnoKV69exaxZs5CdnY2OHTti8+bN+iTLrKwsydLo3bt3x+rVq/Haa6/h1VdfRatWrbBhwwbOMUFERKQgiyoT5a0OKSkp8PHxAQDk5eXh2WefRY8ePTBhwgSMGjUK06ZNw5YtWxQNePLkyZg8eXKV53bs2FHp2LBhwzBs2DBFYyAiIqrEiRMwNUIIs0O/5557sHXr1kqtDr/99hv69u2Lixcv4tChQ+jbty+uXbumWLD2kJ+fD19fX0RjMOpp7k4b7mKlBc4qrUZqYsVRk2pYUdRwim+dn7f0nNGkY8K9+nv98MOXFgRHVLVez02U7DfYa5TobaJLUc4Ko8KKE+1V+8xS9uFYna7i/+sdUYod2Ii8vDz9H8BKKv+eaPbmXLh4eMi6l+72bZyd+XerxWotFiVg5uXl4cqVK5WOX716VT/Jk5+fH0pKqs8XICIiorrBosrE4MGDMW7cOKxfvx4XLlzAhQsXsH79eowfPx5xcXEAgP379+O+++5TMlYiIiKHJXsqbQUSOO3FopyJZcuWYdq0aRgxYgTu/K95sV69eoiPj9dPaBUREYFPP/1UuUiJiIgcGWfANI+Xlxc++eQTLFiwAH/88QcAoHnz5vDy8tKX6dixoyIBOhPjfl+zfqQM8ytq6j82WNrc5eZt6bmSUmkMRsuZC/eK5eaN+7h3LlNwyK1CWqdMkuyH7K74bHZ8+omtwyETjH9+BnaOlewLw59xO+Q5ENXIiRMwLapMlPPy8sIDDzygVCxERESkQhZXJg4ePIivv/4aWVlZlRIt161bJzswIiIiNXHmSassSsBcs2YNunfvjhMnTmD9+vUoLS3Fb7/9hm3btsHX11fpGImIiBwfp9M2z9tvv40FCxYgISEB3t7e+OCDDxAeHo7nnnsOISEhSsfocAzHpWssnQuiNs8xyH2oNAeFMTP6kCVTfBtN4a1xNV2/zH0oVP/a69wtybmWq5+X7J8etbTWMVlL+IZCyX6pt3s1JcnhGP/Mm5iaXg04twTVZRa1TJw5cwYDBw4EcHcBrqKiImg0GkybNk3ZdS+IiIjUQolhoSptmbCoMtGwYUMUFBQAuDsb5rFjxwAAubm5uHnzpqlLiYiI6iZ2c5inZ8+e2Lp1K9q3b49hw4ZhypQp2LZtG7Zu3Yo+ffooHaNDM56K11rdHpWGjdbU7WGKiS4RYfwjUSZtWm6463zFjsEwUQBo+aVRM+4oi6KTxXAVSgBwKZQOfXUvYVOzWpS0CJTsux2/ULEjY2ioPabPJqrrLPpG+vDDD3H79t1/pP/+97/Dzc0Ne/bswdChQ/Haa68pGiAREZEqcJ4J8/j7++tfu7i4YMaMGYoFREREpEbOPDRU1qRVV65cwZUrV6DTSUcEcCIrIiIi52FRZSI9PR3x8fE4ceIEjFcw12g0KHPiPknV5VDU9P/K+L6Gz61X+/cWvmmCZD9km/Ra1+KKn6P/LlkmORf9rPTa/OfzJfsNF1ZM4+5x/XqtY2r3wQuS/WNTPqr1taS8Tm9J/3+Enrsg2VfpH2xETsGib6Bx48bhvvvuw4oVKxAUFASNRp0LkxARESmGORPm+eOPP/DNN9+gZcuWSsdDRESkSs6cM2HRPBN9+vTBL7/8onQsREREpEIWtUx8+umniI+Px7Fjx9CuXTu4uUnnG3jiiScUCa4ucMipt5V6ptE8E0VNvCT7hrkOrXOkk5m5/lkk2dd5e+pfd/ubdNnwRpnXJPv1/2o0BfjN7OqDNMrr0BRXLLEe8EupcWmyo8a/3aq5EJGjU2nLglwWfeukpaVh9+7d+OGHHyqdc/YETCIiclJOnDNhUTfHiy++iP/7v//D5cuXodPpJBsrEkRERM7FopaJ69evY9q0aQgKClI6HiIiIlVy5gRMiyoTQ4YMwfbt29GiRQul41EFw6WENW61/whrWhNAqZwKRdfxMLoXvH0r7ntH+n58DkjnBdD5eetfu+QWSO9jPEdFWcVvUOP/XjSKwehzM47JHAYfxdkhlt+GlLf1yxTJ/oOzpbkzjQ9VzC/icuo8iByOE3dzWPQtc9999yEpKQm7du1C+/btKyVgvvTSS4oER0RERI7P4tEcXl5e2LlzJ3bu3Ck5p9FoWJkgIiKnw24OM2VmZiodB8F0N4icLhDjbg/JfY27QIxjMH5uicFwSqP7ijKjIZs5FVNbG/9+aO5Ic3/LQhrqX7tc+9Pkfc2hgdF7N4i5zfwbklP9lo2W7G/+9guLn0vmM14+Puj6Zcm+4fBhNTDsDiUnwW6OmiUmJuLNN99EgwYNkJiYWG05jUaD9957T5HgiIiIyPHVujJx+PBhlJaW6l9Xh+t0EBGRU2LLRM22b99e5WsiIiKyf87EP/7xDyQlJWHKlClYuHAhAOD27duYPn061qxZg+LiYsTGxuKjjz5SfGoH28y7XNfoDPMKbDR1tRmTgZmTXyFnGKkoLql12crPle7XO5lVcc7cm5mIuVIeh2tFroamRDqddpmnN8i2urxWMfwz8HKW9KTxz+afeRY9w5zfHVI5nZ3/X9uxZeLAgQNYtmwZHnjgAcnxadOm4bvvvsPatWvh6+uLyZMnY8iQIdi9e7fMQKUsmgGTiIiIHENhYSFGjx6NTz75BA0bViSz5+XlYcWKFXj//ffx6KOPIjIyEikpKdizZw/27t2raAysTBARESlBKLSZKSEhAQMHDkRMTIzkeHp6OkpLSyXHIyIi0KRJE6SlpZn/IBPYzUFERKQAJXMm8vPzJce1Wi20Wm2l8mvWrMGhQ4dw4MCBSueys7Ph7u4OPz8/yfGgoCBkZ5tYbdkCrEzIZDyW3Jzpta1FTn5FpTkpjPcN711Tboa1+qqNn2vG9NrC4Ede+DaQnMtrKZ3H4P4lL+hfh6QVS86l/mtFrZ9JFR4bPkayH3jJYC6JGuYtcXScV4KUFBYWJtmfPXs2Xn/9dcmx8+fPY8qUKdi6dSs8PDxsGF1lqunmuHHjBkaPHg0fHx/4+flh/PjxKCwsNHlNdHQ0NBqNZHv++edtFDERETkVBbs5zp8/j7y8PP2WlJRU6XHp6em4cuUKOnfujHr16qFevXrYuXMnFi1ahHr16iEoKAglJSXIzc2VXJeTk4Pg4GBF37r9/4yupdGjR+Py5cvYunUrSktLMXbsWEycOBGrV682ed2ECRPwxhtv6Pfr169v7VCJiMgJKdnN4ePjAx8fH5Nl+/Tpg6NHj0qOjR07FhEREfjb3/6GsLAwuLm5ITU1FUOHDgUAZGRkICsrC1FRUfICNaKKysSJEyewefNmHDhwAA8++CAAYPHixRgwYADmz5+P0NDQaq+tX7++4jWwusTclUwl3SBKrk5qDlMx19T1YhCz5tI1yamA69I+ygD3igXshFa6mN2E8w9L9j8JU3aYVV219euVkv0+/zde/9rjuHTVWZNdbEQEb29vtGvXTnKsQYMGaNSokf74+PHjkZiYCH9/f/j4+ODFF19EVFQUHnroIUVjUUU3R1paGvz8/PQVCQCIiYmBi4sL9u3bZ/LaL774Ao0bN0a7du2QlJSEmzdvmixfXFyM/Px8yUZERFQjO43mMGXBggV4/PHHMXToUPTs2RPBwcFYt26dsg+BSlomsrOzERgYKDlWr149+Pv7m8xIHTVqFJo2bYrQ0FD8+uuv+Nvf/oaMjAyTH2RycjLmzJmjWOxEROQkHGA67R07dkj2PTw8sGTJEixZskTejWtg18rEjBkzMG/ePJNlTpw4YfH9J06cqH/dvn17hISEoE+fPjhz5gxatGhR5TVJSUmShczy8/MrZdUSERFRBbtWJqZPn44xY8aYLNO8eXMEBwfjypUrkuN37tzBjRs3zMqH6NatGwDg9OnT1VYmqhvLW1uOOFRUDnOWRTdrqXNrqWkJdUPG8bqa6PUzypnYtqe99PxTzJkApDkQQM1DaF1vGfw/MP4ZkTNdO/MryA40/9vk3kON7PpNFxAQgICAgBrLRUVFITc3F+np6YiMjAQAbNu2DTqdTl9BqI0jR44AAEJCQiyKl4iIqFoO0M1hL6pIwGzTpg369euHCRMmYP/+/di9ezcmT56MESNG6EdyXLx4EREREdi/fz8A4MyZM3jzzTeRnp6Os2fP4j//+Q+eeeYZ9OzZs9JCKERERHKVDw2Vu6mRKioTwN1RGREREejTpw8GDBiAHj16YPny5frzpaWlyMjI0I/WcHd3x08//YS+ffsiIiIC06dPx9ChQ/Htt9/a6y0QERHVSarp0Pf39zc5QVWzZs0gREWVLiwsDDt37rR+YMZL3rrUfvlvtTPulza19LmpfArAijkVZuRQVFqu3ERI/keNejafku62f79iKu5Gx6VLne/49JPqb6xCzdc9p38dcfmGybK9x02Q7IuKBQ7hWiBdAl5jnDNh8P+ypp8nW+EU2nZm7yXHjTlxN4dqKhNEREQOT6WVAblU081BREREjoktE1Zm2Ayq9mGiNTFnGGmla81otpbVJWLOqqcGcjs0Mnm+79B4yb7oWfG6/mlp0/+OW9I6/Nitz+pfu1+XxpQx9uNax2gthbrbkv2uHydK9tt8ealip0TapTOwR5xkv74uR7IvDKYs1xhda87PhK2GgrJbg0xRcm0Otanb325ERES24sQ5E+zmICIiIlnYMkFERKQAdnMQWZk5w0hrvJdS03bX1M/u3UD/0vdYruTUqSRPyX7gjjzJftMvDJY3ryd9r397/TnJfpuDFWVv3+MjOdf/65HSkOu761//+M2qagKvWre/TdK/bniyUHJOc0uarwDXiqGvLgW3JKfuCZPu42bFvvHw2pqmxNbcMRju6V1fevJGrmTXUYaDElWL3RxERERElmHLBBERkQLYzUFERETyOHE3BysTSjMxvXZdW55cDiVzKCT3lTNfhVFM4s+KPAhNQZHkXOvppnsIJbkDxdJzjbf+IT1gEIfnaWlhw3kYAKCwja/J55qyb17FnBUTzj8sOXdppNHqvYZzPhh9pm7HpZ+FnFwGyb+bRTdrfV/OK+GkHG36bGNOXJlgzgQRERHJ4rx/GhMRESmIORNEREQkjxN3c7AyYUfMoahg2AeuVP5Ejc+saVl0k9ea8SCj3IxKS53DRG6AbwPJ/p73l5rx4Op9ErZbsj8QcdICBp9NpbkjzMmRqCm3gcuKE9UJzvvtRUREpCCNENAIeU0Lcq+3F1YmiIiIlMBuDrIaw6FMLjUsw81uDwA1D/uzRzeIqS4PAKaXM6+h+V4Y/Boad3lobkmno34koWIq7v8uWVZTVBaTdG3U1P0gY5imIy4zTg7E0YeCkp5zflsREREpjKM5iIiISB4n7ubgpFVEREQkC1smHJhhDoWz5k9UxVpTcZt8poxhpJWYitd4im9XaX3fraAijubrpEuZ/zHEdA5F5zcqliAPSsuVnNNojd5fgYkb2ShHwlY4FJSUwm4OIiIikseJuzlYmSAiIlKAM7dMMGeCiIiIZGHLBKmePabirhSDUS5ApeXNDZnIOTD+o0RTT/p+PLJy9a8jlkj/FmhVNEmyf+rpjyX7hnkSLnnSZcQrxWH4fszIkVAyJ8Ja80owR4Ksht0cREREJJdauynkYjcHERERycKWCVsynhq2hum1DXGq7dqxx7DRKuMwnIrbVJcHYLrbo+iWZF9zu7hix0MrOddyda5kv/f2CZJ9z1s3KnYM74MqVgY1g1JdG+zWIADqnkJbiLub3HuoEL+RiIiIFMDRHEREREQWYssEERGREjiag9TGVD8w8ykqOEIOhVLDRgGj5cqN8h5cXIwaGkO8pM8tKa24j3GOhHHeg6k8Dg7/JKqSRnd3k3sPNWI3BxEREcnCP2GJiIiU4MTdHKppmZg7dy66d++O+vXrw8/Pr1bXCCEwa9YshISEwNPTEzExMTh16pR1AyUiIqdUPppD7qZGqmmZKCkpwbBhwxAVFYUVK1bU6pp33nkHixYtwqpVqxAeHo6ZM2ciNjYWx48fh4eHh5UjrgUZ806Ywjkpqmeqj95ec1IoReftKdmvfzJHsl/YPlj/2uu3K5Jz4s+8au/riDkSzImoQ9Q8r4QxzjPh+ObMmQMAWLlyZa3KCyGwcOFCvPbaaxg8eDAA4PPPP0dQUBA2bNiAESNGWCtUIiIip6Kabg5zZWZmIjs7GzExMfpjvr6+6NatG9LS0qq9rri4GPn5+ZKNiIioJuzmqIOys7MBAEFBQZLjQUFB+nNVSU5O1reC2By7PezKVsNIFRsq6u0rvc+tEul5o6GjDfZmVuxo3U0+xxGnyGbXRh1Rl7o1jDEB0z5mzJgBjUZjcjt58qRNY0pKSkJeXp5+O3/+vE2fT0REpDZ2/RN1+vTpGDNmjMkyzZs3t+jewcF3k81ycnIQEhKiP56Tk4OOHTtWe51Wq4VWq632PBERUVWceW0Ou1YmAgICEBAQYJV7h4eHIzg4GKmpqfrKQ35+Pvbt24dJkyZZ5ZlEROTEOJrD8WVlZeHGjRvIyspCWVkZjhw5AgBo2bIlvLzuThscERGB5ORkPPnkk9BoNJg6dSreeusttGrVSj80NDQ0FHFxcfZ7Iw6AU3HXjq2GkZq1XLkhnXTeXU2+dLlyUWyUQ2HwfkSx0RLkzJEgIhlU880xa9YsrFq1Sr/fqVMnAMD27dsRHR0NAMjIyEBeXsV4+VdeeQVFRUWYOHEicnNz0aNHD2zevNkx5pggIqI6xZm7OTRCqLRNxUby8/Ph6+uLaAxGPY2bbR+u0GgOc7BlonasNdKjxpYJg+dqfL2l54xGb5hqmTDGlgmyGTuM5rgjSrEDG5GXlwcfHx/F71/+PRHV7w3Uc5P3x+qd0ttI2zzLarFaS52dZ4KIiIhsg3+GOjLDGryNWik4J0Xt1PQXuKUtFzXNQaExmB9CFElzJGy1jDhbH8hsdXluCQPO3M3BbwoiIiIl6MTdTe49VIiVCSIiIiVwBkwiIiIiy7BlQi2stG5HTTgnhWXMySswlV9hnNsg8gssjskcXCqcZHGSHAljGiiQM6FIJLbHbwMiIiIlOPEMmOzmICIiIlnYMkEWq6kJm90gtaPkUEt7YFcG0V0cGkpERETycDQHERERkWVYmSAiIlKARghFNnMkJyejS5cu8Pb2RmBgIOLi4pCRkSEpc/v2bSQkJKBRo0bw8vLC0KFDkZOTo+RbZ2VCtXRl0s0BidI7td7IsfD/HZlFBf8e2YROoc0MO3fuREJCAvbu3YutW7eitLQUffv2RVFRkb7MtGnT8O2332Lt2rXYuXMnLl26hCFDhsh7r0aYM0FERKRSmzdvluyvXLkSgYGBSE9PR8+ePZGXl4cVK1Zg9erVePTRRwEAKSkpaNOmDfbu3YuHHnpIkTjYMkFERKQAJbs58vPzJVtxcXGtYsjLywMA+Pv7AwDS09NRWlqKmJgYfZmIiAg0adIEaWlpir13ViaIiIiUIBTaAISFhcHX11e/JScn1/h4nU6HqVOn4uGHH0a7du0AANnZ2XB3d4efn5+kbFBQELKzs2W+4Qrs5qgr7LBcuZLM6Xvn/BWWYX4DKcqZcyOqo+AMmOfPn4ePj4/+sFarrfHShIQEHDt2DLt27ZIXgwX4rzIREZGD8fHxkVQmajJ58mRs2rQJP//8M+6991798eDgYJSUlCA3N1fSOpGTk4Pg4GDF4mU3BxERkQLKZ8CUu5lDCIHJkydj/fr12LZtG8LDwyXnIyMj4ebmhtTUVP2xjIwMZGVlISoqSom3DYAtE3WTnVYYtRVrNdc7QvcJuyLIYbFbo2Z2WOgrISEBq1evxsaNG+Ht7a3Pg/D19YWnpyd8fX0xfvx4JCYmwt/fHz4+PnjxxRcRFRWl2EgOgJUJIiIi1fr4448BANHR0ZLjKSkpGDNmDABgwYIFcHFxwdChQ1FcXIzY2Fh89NFHisbBygQREZECNLq7m9x7mEPUoiXDw8MDS5YswZIlSyyMqmasTBARESnBDt0cjoKVCWdQx3MolMJ8BSIDzJEgM7AyQUREpAQnXoKclQkiIiIFWLLqZ1X3UCPOM0FERESysGXCGal86m0isgLmSMjHBEwiIiKSRQCQOTSUORNEREROzJlzJliZcHYcNkrkvNi1QQphZYKIiEgJAgrkTCgSic2xMkFERKQEJ07A5NBQIiIikoUtEyTFHAqiuos5EtalA6BR4B4qpJqWiblz56J79+6oX78+/Pz8anXNmDFjoNFoJFu/fv2sGygRETml8tEccjc1Uk3LRElJCYYNG4aoqCisWLGi1tf169cPKSkp+n2tVmuN8IiIiJyWaioTc+bMAQCsXLnSrOu0Wi2Cg4OtEBEREZEBJmDWXTt27EBgYCBat26NSZMm4fr16/YOSV10ZdKNiNSDv7+2VV6ZkLupkGpaJizRr18/DBkyBOHh4Thz5gxeffVV9O/fH2lpaXB1rTqxsLi4GMXFxfr9/Px8W4VLRESkSnZtmZgxY0alBEnj7eTJkxbff8SIEXjiiSfQvn17xMXFYdOmTThw4AB27NhR7TXJycnw9fXVb2FhYRY/n4iInAhbJuxj+vTpGDNmjMkyzZs3V+x5zZs3R+PGjXH69Gn06dOnyjJJSUlITEzU7+fn57NCQURENXPioaF2rUwEBAQgICDAZs+7cOECrl+/jpCQkGrLaLVajvgwhcuXEzk25kbYjTMv9KWaBMysrCwcOXIEWVlZKCsrw5EjR3DkyBEUFhbqy0RERGD9+vUAgMLCQrz88svYu3cvzp49i9TUVAwePBgtW7ZEbGysvd4GERFRnaOaBMxZs2Zh1apV+v1OnToBALZv347o6GgAQEZGBvLy8gAArq6u+PXXX7Fq1Srk5uYiNDQUffv2xZtvvsmWByIiUp4TDw1VTWVi5cqVNc4xIQz+J3h6emLLli1WjsrJceptIvtjt4bj0AlAI7MyoFNnZUI13RxERETkmFTTMkFEROTQ2M1BRERE8igxTwQrE+TsTPXdMp+CyHLMiyAHx8oEERGREtjNQURERLLoBGR3U3A0BxERETkjtkyQbXBOCqLaY46EOgnd3U3uPVSIlQkiIiIlMGeCiIiIZHHinAlWJsg+OIyUnBm7MaiOYWWCiIhICezmICIiIlkEFKhMKBKJzXFoKBEREcnClglyPBxGSnUR8yTqPnZzEBERkSw6HQCZ80To1DnPBLs5iIiISBa2TBARESmB3RxEDqymvmbmVJAjYE4EOXFlgt0cREREJAtbJoiIiJTA6bSJVMyc5mV2iZA52HVBZhBCByFz1U+519sLKxNERERKEEJ+ywJzJoiIiMgZsWWCiIhICUKBnAmVtkywMkHOhUufkzHmRZBSdDpAIzPnQaU5E+zmICIiIlnYMkFERKQEdnMQERGRHEKng5DZzcGhoURqx/kq6g7mQRDZFCsTRERESmA3BxEREcmiE4DGOSsTHM1BREREsrBlgsgScvrkmW9RO8x7ILURAoDceSbU2TLBygQREZEChE5AyOzmECqtTKiim+Ps2bMYP348wsPD4enpiRYtWmD27NkoKSkxed3t27eRkJCARo0awcvLC0OHDkVOTo6NoiYiIqcidMpsKqSKlomTJ09Cp9Nh2bJlaNmyJY4dO4YJEyagqKgI8+fPr/a6adOm4bvvvsPatWvh6+uLyZMnY8iQIdi9e7cNoycywuZ7IqpjVFGZ6NevH/r166ffb968OTIyMvDxxx9XW5nIy8vDihUrsHr1ajz66KMAgJSUFLRp0wZ79+7FQw89ZJPYiYjIObCbQ4Xy8vLg7+9f7fn09HSUlpYiJiZGfywiIgJNmjRBWlqaLUIkIiJnwm4OdTl9+jQWL15ssosjOzsb7u7u8PPzkxwPCgpCdnZ2tdcVFxejuLhYv5+XlwcAuINS2XOREBGR7d1BKQDr/9WvxPdEeaxqY9fKxIwZMzBv3jyTZU6cOIGIiAj9/sWLF9GvXz8MGzYMEyZMUDym5ORkzJkzp9LxXfhe8WcREZHtXL9+Hb6+vorf193dHcHBwdiVrcz3RHBwMNzd3RW5l61ohB07aK5evYrr16+bLNO8eXP9h3rp0iVER0fjoYcewsqVK+HiUn0vzbZt29CnTx/8+eefktaJpk2bYurUqZg2bVqV1xm3TOTm5qJp06bIysqyyg+hteTn5yMsLAznz5+Hj4+PvcOpNbXGDag3dsZtW4zb9vLy8tCkSZNK3wdKun37do0jDGvL3d0dHh4eitzLVuzaMhEQEICAgIBalb148SJ69+6NyMhIpKSkmKxIAEBkZCTc3NyQmpqKoUOHAgAyMjKQlZWFqKioaq/TarXQarWVjvv6+qruFwgAfHx8GLeNqTV2xm1bjNv2avrekMPDw0N1FQAlqSIB8+LFi4iOjkaTJk0wf/58XL16FdnZ2ZLch4sXLyIiIgL79+8HcPfLf/z48UhMTMT27duRnp6OsWPHIioqiiM5iIiIFKSKBMytW7fi9OnTOH36NO69917JufJemtLSUmRkZODmzZv6cwsWLICLiwuGDh2K4uJixMbG4qOPPrJp7ERERHWdKioTY8aMwZgxY0yWadasWaVMXQ8PDyxZsgRLliyx+NlarRazZ8+usuvDkTFu21Nr7Izbthi37ak5drWwawImERERqZ8qciaIiIjIcbEyQURERLKwMkFERESysDJBREREsrAyYeTs2bMYP348wsPD4enpiRYtWmD27Nk1zmx2+/ZtJCQkoFGjRvDy8sLQoUORk5Njo6jvmjt3Lrp374769evXepa3MWPGQKPRSDbDFVptwZK4hRCYNWsWQkJC4OnpiZiYGJw6dcq6gRq5ceMGRo8eDR8fH/j5+WH8+PEoLCw0eU10dHSlz/v555+3eqxLlixBs2bN4OHhgW7duunnY6nO2rVrERERAQ8PD7Rv3x7ff2+f6eTNiXvlypWVPlt7TCL0888/Y9CgQQgNDYVGo8GGDRtqvGbHjh3o3LkztFotWrZsiZUrV1o9TmPmxr1jx45Kn7dGozG59pE1JCcno0uXLvD29kZgYCDi4uKQkZFR43WO8jNeV7AyYeTkyZPQ6XRYtmwZfvvtNyxYsABLly7Fq6++avK6adOm4dtvv8XatWuxc+dOXLp0CUOGDLFR1HeVlJRg2LBhmDRpklnX9evXD5cvX9ZvX375pZUirJolcb/zzjtYtGgRli5din379qFBgwaIjY3F7du3rRip1OjRo/Hbb79h69at2LRpE37++WdMnDixxusmTJgg+bzfeecdq8b51VdfITExEbNnz8ahQ4fQoUMHxMbG4sqVK1WW37NnD0aOHInx48fj8OHDiIuLQ1xcHI4dO2bVOOXGDdydndHwsz137pwNI76rqKgIHTp0qPWQ9MzMTAwcOBC9e/fGkSNHMHXqVDz77LPYsmWLlSOVMjfuchkZGZLPPDAw0EoRVm3nzp1ISEjA3r17sXXrVpSWlqJv374oKiqq9hpH+RmvUwTV6J133hHh4eHVns/NzRVubm5i7dq1+mMnTpwQAERaWpotQpRISUkRvr6+tSobHx8vBg8ebNV4aqu2cet0OhEcHCzeffdd/bHc3Fyh1WrFl19+acUIKxw/flwAEAcOHNAf++GHH4RGoxEXL16s9rpevXqJKVOm2CDCCl27dhUJCQn6/bKyMhEaGiqSk5OrLD98+HAxcOBAybFu3bqJ5557zqpxGjM3bnN+7m0FgFi/fr3JMq+88opo27at5NhTTz0lYmNjrRiZabWJe/v27QKA+PPPP20SU21duXJFABA7d+6stoyj/IzXJWyZqIW8vDz4+/tXez49PR2lpaWIiYnRH4uIiECTJk2QlpZmixBl2bFjBwIDA9G6dWtMmjSpxsXX7C0zMxPZ2dmSz9vX1xfdunWz2eedlpYGPz8/PPjgg/pjMTExcHFxwb59+0xe+8UXX6Bx48Zo164dkpKSJLO2Kq2kpATp6emSz8rFxQUxMTHVflZpaWmS8gAQGxtr059lS+IGgMLCQjRt2hRhYWEYPHgwfvvtN1uEK4sjfN5ydOzYESEhIXjsscewe/due4eDvLw8ADD5b7baP3NHpIoZMO3p9OnTWLx4MebPn19tmezsbLi7u1fq7w8KCrJ5/6G5+vXrhyFDhiA8PBxnzpzBq6++iv79+yMtLQ2urq72Dq9K5Z9pUFCQ5LgtP+/s7OxKzbn16tWDv7+/yRhGjRqFpk2bIjQ0FL/++iv+9re/ISMjA+vWrbNKnNeuXUNZWVmVn9XJkyervCY7O9uuny1gWdytW7fGZ599hgceeAB5eXmYP38+unfvjt9++63SNPyOpLrPOz8/H7du3YKnp6edIjMtJCQES5cuxYMPPoji4mJ8+umniI6Oxr59+9C5c2e7xKTT6TB16lQ8/PDDaNeuXbXlHOFnvK5xmpaJGTNmVJksZLgZ/yN18eJF9OvXD8OGDcOECRNUE7c5RowYgSeeeALt27dHXFwcNm3ahAMHDmDHjh0OHbe1WDvuiRMnIjY2Fu3bt8fo0aPx+eefY/369Thz5oyC78I5RUVF4ZlnnkHHjh3Rq1cvrFu3DgEBAVi2bJm9Q6uTWrdujeeeew6RkZHo3r07PvvsM3Tv3h0LFiywW0wJCQk4duwY1qxZY7cYnJXTtExMnz69xvU9mjdvrn996dIl9O7dG927d8fy5ctNXhccHIySkhLk5uZKWidycnIQHBwsJ2yz45arefPmaNy4MU6fPo0+ffpYfB9rxl3+mebk5CAkJER/PCcnBx07drTonuVqG3dwcHClRMA7d+7gxo0bZv0/79atG4C7LWAtWrQwO96aNG7cGK6urpVGFpn62QwODjarvDVYErcxNzc3dOrUCadPn7ZGiIqp7vP28fFx2FaJ6nTt2hW7du2yy7MnT56sT4SuqSXKEX7G6xqnqUwEBAQgICCgVmUvXryI3r17IzIyEikpKXBxMd2AExkZCTc3N6SmpmLo0KEA7mY4Z2VlISoqymZxK+HChQu4fv265EvaEtaMOzw8HMHBwUhNTdVXHvLz87Fv3z6zR7IYq23cUVFRyM3NRXp6OiIjIwEA27Ztg06n01cQauPIkSMAIPvzro67uzsiIyORmpqKuLg4AHebglNTUzF58uQqr4mKikJqaiqmTp2qP7Z161bZP8vmsCRuY2VlZTh69CgGDBhgxUjli4qKqjQs0daft1KOHDlitZ/l6ggh8OKLL2L9+vXYsWMHwsPDa7zGEX7G6xx7Z4A6mgsXLoiWLVuKPn36iAsXLojLly/rN8MyrVu3Fvv27dMfe/7550WTJk3Etm3bxMGDB0VUVJSIioqyaeznzp0Thw8fFnPmzBFeXl7i8OHD4vDhw6KgoEBfpnXr1mLdunVCCCEKCgrEX//6V5GWliYyMzPFTz/9JDp37ixatWolbt++7bBxCyHEP/7xD+Hn5yc2btwofv31VzF48GARHh4ubt26ZbO4+/XrJzp16iT27dsndu3aJVq1aiVGjhypP2/8c3L69GnxxhtviIMHD4rMzEyxceNG0bx5c9GzZ0+rxrlmzRqh1WrFypUrxfHjx8XEiROFn5+fyM7OFkII8fTTT4sZM2boy+/evVvUq1dPzJ8/X5w4cULMnj1buLm5iaNHj1o1Trlxz5kzR2zZskWcOXNGpKenixEjRggPDw/x22+/2TTugoIC/c8wAPH++++Lw4cPi3PnzgkhhJgxY4Z4+umn9eX/+OMPUb9+ffHyyy+LEydOiCVLlghXV1exefNmh457wYIFYsOGDeLUqVPi6NGjYsqUKcLFxUX89NNPNo170qRJwtfXV+zYsUPy7/XNmzf1ZRz1Z7wuYWXCSEpKigBQ5VYuMzNTABDbt2/XH7t165Z44YUXRMOGDUX9+vXFk08+KamA2EJ8fHyVcRvGCUCkpKQIIYS4efOm6Nu3rwgICBBubm6iadOmYsKECfp/rB01biHuDg+dOXOmCAoKElqtVvTp00dkZGTYNO7r16+LkSNHCi8vL+Hj4yPGjh0rqQAZ/5xkZWWJnj17Cn9/f6HVakXLli3Fyy+/LPLy8qwe6+LFi0WTJk2Eu7u76Nq1q9i7d6/+XK9evUR8fLyk/Ndffy3uu+8+4e7uLtq2bSu+++47q8dYFXPinjp1qr5sUFCQGDBggDh06JDNYy4fMmm8lccaHx8vevXqVemajh07Cnd3d9G8eXPJz7qjxj1v3jzRokUL4eHhIfz9/UV0dLTYtm2bzeOu7t9rw8/QkX/G6wouQU5ERESyOM1oDiIiIrIOViaIiIhIFlYmiIiISBZWJoiIiEgWViaIiIhIFlYmiIiISBZWJoiIiEgWViaIqFbOnj0LjUajnwKciKgcKxNEREQkCysTRE6gpKTE3iEQUR3GygRRHRQdHY3Jkydj6tSpaNy4MWJjY3Hs2DH0798fXl5eCAoKwtNPP41r167pr9m8eTN69OgBPz8/NGrUCI8//jjOnDljx3dBRGrBygRRHbVq1Sq4u7tj9+7d+Mc//oFHH30UnTp1wsGDB7F582bk5ORg+PDh+vJFRUVITEzEwYMHkZqaChcXFzz55JPQ6XR2fBdEpAZc6IuoDoqOjkZ+fj4OHToEAHjrrbfw3//+F1u2bNGXuXDhAsLCwpCRkYH77ruv0j2uXbuGgIAAHD16FO3atcPZs2cRHh6Ow4cPo2PHjrZ6K0SkAmyZIKqjIiMj9a9/+eUXbN++HV5eXvotIiICAPRdGadOncLIkSPRvHlz+Pj4oFmzZgCArKwsm8dOROpSz94BEJF1NGjQQP+6sLAQgwYNwrx58yqVCwkJAQAMGjQITZs2xSeffILQ0FDodDq0a9eOyZtEVCNWJoicQOfOnfHNN9+gWbNmqFev8q/99evXkZGRgU8++QSPPPIIAGDXrl22DpOIVIrdHEROICEhATdu3MDIkSNx4MABnDlzBlu2bMHYsWNRVlaGhg0bolGjRli+fDlOnz6Nbdu2ITEx0d5hE5FKsDJB5ARCQ0Oxe/dulJWVoW/fvmjfvj2mTp0KPz8/uLi4wMXFBWvWrEF6ejratWuHadOm4d1337V32ESkEhzNQURERLKwZYKIiIhkYWWCiIiIZGFlgoiIiGRhZYKIiIhkYWWCiIiIZGFlgoiIiGRhZYKIiIhkYWWCiIiIZGFlgoiIiGRhZYKIiIhkYWWCiIiIZGFlgoiIiGT5f+u18vuU2OA8AAAAAElFTkSuQmCC\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from matplotlib.font_manager import X11FontDirectories\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "## a)\n",
        "a = np.float(input('Value of a:'))\n",
        "b = np.float(input('Value of b:'))\n",
        "c = np.float(input('Value of a:'))\n",
        "\n",
        "x1= (-b+np.sqrt(b**2-4*a*c))/(2*a)\n",
        "x2= (-b-np.sqrt(b**2-4*a*c))/(2*a)\n",
        "print(x1, x2)\n",
        "\n",
        "## b)\n",
        "X1 = (2*c)/(-b-np.sqrt(b**2-4*a*c))\n",
        "X2 = (2*c)/(-b+np.sqrt(b**2-4*a*c))\n",
        "print(X1, X2)\n",
        "## c)\n",
        "xn1 = x1\n",
        "xn2 = X2\n",
        "print(xn1, xn2)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "H0vw0WGmbAn_",
        "outputId": "5a092238-0243-48ed-d47d-b675c2da3448"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "<ipython-input-5-a4dd9cfd5359>:5: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.\n",
            "Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations\n",
            "  a = np.float(input('Value of a:'))\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Value of a:0.001\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "<ipython-input-5-a4dd9cfd5359>:6: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.\n",
            "Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations\n",
            "  b = np.float(input('Value of b:'))\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Value of b:1000\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "<ipython-input-5-a4dd9cfd5359>:7: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.\n",
            "Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations\n",
            "  c = np.float(input('Value of a:'))\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Value of a:0.001\n",
            "-9.999894245993346e-07 -999999.999999\n",
            "-1.000000000001e-06 -1000010.5755125057\n",
            "-9.999894245993346e-07 -1000010.5755125057\n"
          ]
        }
      ]
    }
  ]
}