#include <iostream>
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui.hpp>

using namespace cv;

/** --- FILTRE MOYENNEUR --- **/
Mat filtreM(Mat input) {
    Mat output;
    Mat kernel = (Mat_<float>(3, 3) <<
                                    1.0 / 16, 2.0 / 16, 1.0 / 16,
            2.0 / 16, 4.0 / 16, 2.0 / 16,
            1.0 / 16, 2.0 / 16, 1.0 / 16);

    // Appliquer le filtrage
    filter2D(input, output, -1, kernel);

    return output;
}

/** --- MEDIANE --- **/
Mat medianBlur(Mat input) {
    Mat output;
    medianBlur(input, output, 3);
    return output;
}

/** --- REHAUSSEMENT DE CONTRASTE --- **/
Mat rehaussementContraste(Mat input, int alpha) {
    Mat output;

    Mat matrice = (Mat_<float>(3, 3) << 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
    Mat laplacien = (Mat_<float>(3, 3) << 0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0);
    Mat result = matrice - alpha * laplacien;

    filter2D(input, output, -1, result);
    return output;
}

/** --- FILTRES DERIVATIFS --- **/
Mat sobelX(Mat input, float delta = 128.0) {
    Mat output;

    Mat kernel = (Mat_<float>(3, 3) <<
                                    -1.0 / 4.0, 0.0, 1.0 / 4.0,
            -2.0 / 4.0, 0.0, 2.0 / 4.0,
            -1.0 / 4.0, 0.0, 1.0 / 4.0);

    filter2D(input, output, -1, kernel, Point(-1, -1), delta);
    return output;
}

Mat sobelY(Mat input, float delta = 128.0) {
    Mat output;

    Mat kernel = (Mat_<float>(3, 3) <<
                                    -1.0 / 4.0, -2.0 / 4.0, -1.0 / 4.0,
            0.0, 0.0, 0.0,
            1.0 / 4.0, 2.0 / 4.0, 1.0 / 4.0);

    filter2D(input, output, -1, kernel, Point(-1, -1), delta);
    return output;
}

/** --- GRADIENT --- **/
Mat gradientFromSobel(Mat input) {
    Mat output;
    Mat imageSobelX;
    Mat imageSobelY;

    input.convertTo(input, CV_32FC1);

    input.copyTo(output);
    input.copyTo(imageSobelX);
    input.copyTo(imageSobelY);

    imageSobelX = sobelX(imageSobelX, 0.0);
    imageSobelY = sobelY(imageSobelY, 0.0);

    int nbCols = input.cols;
    int nbRows = input.rows;

    for (int y = 0; y < nbCols; y++) {
        for (int x = 1; x < nbRows; x++) {
            output.at<float>(x, y) = sqrt(
                    (imageSobelX.at<float>(x, y) * imageSobelX.at<float>(x, y))
                    + (imageSobelY.at<float>(x, y) * imageSobelY.at<float>(x, y)));
        }
    }

    input.convertTo(input, CV_8UC1);
    output.convertTo(output, CV_8UC1);

    return output;
}

/** --- DETECTION MARR-HILDRETH --- **/
bool isChangedInNeighborhood(Mat input, Mat laplacien, int x, int y) {
    for (int k = x - 1; k < x + 2; k++) {
        for (int n = y - 1; n < y + 2; n++) {
            if ((input.at<float>(k, n) < 0 && laplacien.at<float>(k, n) >= 0)
                || (input.at<float>(k, n) >= 0 && laplacien.at<float>(k, n) < 0)) {
                return true;
            }
        }
    }
    return false;
}

Mat seuilMarrHildreth(Mat input, int seuil, int alpha) {
    Mat output;
    Mat imageGradient;
    Mat imageLaplacien;

    input.convertTo(input, CV_32FC1);

    input.copyTo(output);
    input.copyTo(imageGradient);
    input.copyTo(imageLaplacien);

    imageLaplacien = rehaussementContraste(input, alpha);

    imageGradient = gradientFromSobel(imageGradient);
    imageGradient.convertTo(imageGradient, CV_32FC1);

    for (int y = 0; y < input.cols - 1; y++) {
        for (int x = 1; x < input.rows - 1; x++) {
            bool isChanged = isChangedInNeighborhood(input, imageLaplacien, x, y);
            if (imageGradient.at<float>(x, y) >= (float) seuil && isChanged) {
                output.at<float>(x, y) = 0.0;
            } else {
                output.at<float>(x, y) = 255.0;
            }
        }
    }

    input.convertTo(input, CV_8UC1);
    output.convertTo(output, CV_8UC1);

    return output;
}

/** --- ESQUISSE --- **/
double rand01() {
    return rand() / (double) RAND_MAX;
}

Mat esquisse(Mat input, int seuil, int alpha, int proportion, int longueur) {
    Mat output;
    Mat imageGradient;
    Mat imageLaplacien;

    input.convertTo(input, CV_32FC1);

    input.copyTo(output);
    input.copyTo(imageGradient);
    input.copyTo(imageLaplacien);

    imageLaplacien = rehaussementContraste(input, alpha);

    imageGradient = gradientFromSobel(imageGradient);
    imageGradient.convertTo(imageGradient, CV_32FC1);

    for (int y = 0; y < input.cols; y++) {
        for (int x = 1; x < input.rows; x++) {
            bool isChanged = isChangedInNeighborhood(input, imageLaplacien, x, y);
            if (imageGradient.at<float>(x, y) >= (float) seuil && isChanged) {
                if (rand01() < (proportion / 100.0)) {
                    double theta = atan2(-y, x) + M_PI / 2 + 0.02 * (rand01() - 0.5);
                    float g = imageGradient.at<float>(x, y);
                    double longueurP = (g / 255.0) * (longueur / 100.0);
                    line(output,
                         Point_<float>(y + longueurP * cos(theta), x + longueurP * sin(theta)),
                         Point_<float>(y - longueurP * cos(theta), x - longueurP * sin(theta)),
                         0, 1, 1);
                } else {
                    output.at<float>(x, y) = 255.0;
                }
            } else {
                output.at<float>(x, y) = 255.0;
            }
        }
    }

    input.convertTo(input, CV_8UC1);
    output.convertTo(output, CV_8UC1);

    return output;
}

/** --- MAIN --- **/
int main(int, char *argv[]) {
    if (argv[1] == nullptr) {
        std::cout << "\nUsage : ./main_t2 <nom-fichier-image>\n"
                  << std::endl;
        exit(1);
    }

    String filename = argv[1];
    String path = "/home/leodie/Documents/elodie/projects/TP1/";

    namedWindow("TP2 - Image");               // cr??e une fen??tre
    Mat input = imread(path + argv[1]);     // lit l'image donn??e en param??tre

    // Trackbar pour le rehausseur
    int alpha = 20;

    createTrackbar("alpha (en %)", "TP2 - Image", nullptr, 200, nullptr);
    setTrackbarPos("alpha (en %)", "TP2 - Image", alpha);

    // Trackbar pour Marr-Hildreth
    int seuil = 20;

    createTrackbar("seuil (en %)", "TP2 - Image", nullptr, 200, nullptr);
    setTrackbarPos("seuil (en %)", "TP2 - Image", seuil);

    // Trackbar pour esquisse
    int proportion = 50;
    int longueur = 100;

    createTrackbar("proportion (en %)", "TP2 - Image", nullptr, 100, nullptr);
    setTrackbarPos("proportion (en %)", "TP2 - Image", proportion);
    createTrackbar("longueur (en %)", "TP2 - Image", nullptr, 1000, nullptr);
    setTrackbarPos("longueur (en %)", "TP2 - Image", longueur);

    // Conversion de la photo en noir et blanc
    if (input.channels() == 3)
        cv::cvtColor(input, input, COLOR_BGR2GRAY);

    Mat output = input.clone();

    while (true) {
        int keycode = waitKey(50);
        int asciicode = keycode & 0xff;
        if (asciicode == 'q') break;

        /** --- DEBUT DES APPELS DE FONCTIONS --- **/
        switch (asciicode) {
            case 'a':
                output = filtreM(input);
                break;
            case 'm':
                output = medianBlur(input);
                break;
            case 's':
                // r??cup??re la valeur courante de alpha
                alpha = getTrackbarPos("alpha (en %)", "TP2 - Image");
                output = rehaussementContraste(input, alpha);
                break;
            case 'x':
                output = sobelX(input);
                break;
            case 'y':
                output = sobelY(input);
                break;
            case 'g':
                output = gradientFromSobel(input);
                break;
            case 't':
                // r??cup??re la valeur courante de seuil et alpha
                seuil = getTrackbarPos("seuil (en %)", "TP2 - Image");
                alpha = getTrackbarPos("alpha (en %)", "TP2 - Image");
                output = seuilMarrHildreth(input, seuil, alpha);
                break;
            case 'e':
                // r??cup??re la valeur courante de seuil, de proportion et de longueur
                seuil = getTrackbarPos("seuil (en %)", "TP2 - Image");
                alpha = getTrackbarPos("alpha (en %)", "TP2 - Image");
                proportion = getTrackbarPos("proportion (en %)", "TP2 - Image");
                longueur = getTrackbarPos("longueur (en %)", "TP2 - Image");
                output = esquisse(input, seuil, alpha, proportion, longueur);
            default:
                break;
        }
        /** --- FIN DES APPELS DE FONCTIONS --- **/

        imshow("TP2 - Image", output);            // l'affiche dans la fen??tre
    }

    imwrite("result.png", input);          // sauvegarde le r??sultat
}