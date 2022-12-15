#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;

/** HISTOGRAMME **/
std::vector<double> histogramme(Mat image) {
    std::vector<double> histogramme(256, 0.0);

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            int greyLevel = (int) image.at<uchar>(i, j);
            histogramme[greyLevel]++;
        }
    }

    for (int i = 0; i < histogramme.size(); i++) {
        histogramme[i] = histogramme[i] / (image.rows * image.cols);
    }

    return histogramme;
}

/** HISTOGRAMME CUMULE **/
std::vector<double> histogramme_cumule(const std::vector<double> &h_I) {
    std::vector<double> histogrammeCumule(256, 0.0);

    for (int i = 0; i < h_I.size(); i++) {
        if (i == 0) {
            histogrammeCumule[i] = h_I[i];
        } else {
            histogrammeCumule[i] = h_I[i] + histogrammeCumule[i - 1];
        }
    }

    return histogrammeCumule;
}

/** AFFICHAGE DES DEUX HISTOGRAMMES **/
Mat afficheHistogrammes(const std::vector<double> &h_I,
                            const std::vector<double> &H_I) {
    Mat image(256, 512, CV_8UC1, 255.0);

    // Affichage de l'histogramme
    auto maxValue_h_I = std::max_element(h_I.begin(), h_I.end());

    for (int i = 0; i < h_I.size(); i++) {
        auto newValue = (((h_I[i]) * image.rows) / *maxValue_h_I);
        for (int j = 0; j < newValue; j++) {
            image.at<uchar>(image.rows - 1 - j, i) = 0;
        }
    }

    // Affichage de l'histogramme cumulé
    auto maxValue_H_I = std::max_element(H_I.begin(), H_I.end());

    for (int i = 0; i < H_I.size(); i++) {
        auto newValue = (((H_I[i]) * image.rows) / *maxValue_H_I);
        for (int j = 0; j < newValue; j++) {
            image.at<uchar>(image.rows - 1 - j, h_I.size() + i) = 0;
        }
    }

    return image;
}

/** EGALISATION **/
Mat equalization(Mat image,
                     std::vector<double> &h_I,
                     std::vector<double> &H_I) {
    // Parcours tout les pixels de l'image
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            int greyLevel = (int) image.at<uchar>(i, j);
            image.at<uchar>(i, j) = 255.0 * (double) H_I[greyLevel];
        }
    }
    h_I = histogramme(image);
    H_I = histogramme_cumule(h_I);
    return image;
}

/** TRAMAGE FLOYD STEINBERG **/
void tramage_floyd_steinberg(Mat input, Mat output) {
    input.convertTo(input, CV_32FC1);
    input.copyTo(output);

    // pour chaque x de gauche à droite
    for (int y = 0; y < output.cols - 1; y++) {
        // pour chaque y de haut en bas
        for (int x = 1; x < output.rows - 1; x++) {
            float ancien_pixel = output.at<float>(x, y);
            float nouveau_pixel = (ancien_pixel > 128.0) ? 255.0 : 0.0;
            output.at<float>(x, y) = nouveau_pixel;
            float erreur_quantification = ancien_pixel - nouveau_pixel;
            output.at<float>(x + 1, y) = output.at<float>(x + 1, y) + (7.0 / 16.0 * erreur_quantification);
            output.at<float>(x - 1, y + 1) = output.at<float>(x - 1, y + 1) + (3.0 / 16.0 * erreur_quantification);
            output.at<float>(x, y + 1) = output.at<float>(x, y + 1) + (5.0 / 16.0 * erreur_quantification);
            output.at<float>(x + 1, y + 1) = output.at<float>(x + 1, y + 1) + (1.0 / 16.0 * erreur_quantification);
        }
    }

    input.convertTo(input, CV_8UC1);
    output.convertTo(output, CV_8UC1);
}

/** MAIN **/
int main(int, char *argv[]) {
    if (argv[1] == nullptr || argv[2] == nullptr) {
        std::cout << "\nUsage : ./main_grey_img <nom-fichier-image> <egal | tram | none>\n" << std::endl;
        exit(1);
    }

    String filename = argv[1];
    String path = "/home/leodie/Documents/elodie/projects/TP1/";

    int old_value = 0;
    int value = 128;

    namedWindow("TP1 Grey IMG");               // crée une fenêtre
    createTrackbar("track grey", "TP1 Grey IMG", &value, 255, nullptr); // un slider

    Mat f = imread(path + filename);        // lit l'image  donné en argument

    /** --- DEBUT DES APPELS DE FONCTIONS --- **/
    // Converti l'image en noir et blanc
    if (f.channels() > 1) {
        cvtColor(f, f, COLOR_RGB2GRAY);
    }

    String functionToExecute = argv[2];

    if (functionToExecute == "egal") {
        // Histogrammes
        std::vector<double> hist = histogramme(f);
        std::vector<double> histCumule = histogramme_cumule(hist);

        // Egalisation
        Mat equalizedImg = equalization(f, hist, histCumule);

        // Affichage des histogrammes
        Mat displayHistogrammes = afficheHistogrammes(hist, histCumule);
        namedWindow("Histogrammes Grey IMG");
        imshow("Histogrammes Grey IMG", displayHistogrammes);                // l'affiche dans la fenêtre

        imshow("TP1 Grey IMG", equalizedImg);
    } else if (functionToExecute == "tram") {
        // Tramage Floyd Steinberg
        Mat tramedImg(f.rows, f.cols, CV_32FC1, 0.0);
        tramage_floyd_steinberg(f, tramedImg);
        imshow("TP1 Grey IMG", tramedImg);
    } else if (functionToExecute == "none") {
        imshow("TP1 Grey IMG", f);
    } else {
        std::cout << "\nUsage : ./main_grey_img <nom-fichier-image> <egal | tram | none>\n" << std::endl;
        exit(1);
    }

    /** --- FIN DES APPELS DE FONCTIONS --- **/

    while (waitKey(50) < 0)          // attend une touche
    { // Affiche la valeur du slider
        if (value != old_value) {
            old_value = value;
            std::cout << "value=" << value << std::endl;
        }
    }
}