#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;

/** HISTOGRAMME **/
std::vector<double> histogramme(Mat image) {
    std::vector<double> histogramme(256, 0.0);

    std::vector<Mat> hsv_channels;
    split(image, hsv_channels);
    Mat v_image = hsv_channels[2];

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            int lightLevel = (int) v_image.at<uchar>(i, j);
            histogramme[lightLevel]++;
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
Mat afficheHistogrammes(const std::vector<double> &h_I, const std::vector<double> &H_I) {
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
Mat equalization(Mat image, std::vector<double> &h_I, std::vector<double> &H_I) {
    std::vector<Mat> hsv_channels;
    split(image, hsv_channels);

    // Parcours tout les pixels de l'image
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            int lightLevel = (int) hsv_channels[2].at<uchar>(i, j);
            hsv_channels[2].at<uchar>(i, j) = 255.0 * (double) H_I[lightLevel];
        }
    }

    merge(hsv_channels, image);

    h_I = histogramme(image);
    H_I = histogramme_cumule(h_I);

    return image;
}

/** TRAMAGE FLOYD STEINBERG (pour BGR) **/
void tramage_floyd_steinberg(Mat input, Mat output) {
    input.convertTo(input, CV_32FC3);

    std::vector<Mat> channels;
    split(input, channels);

    for (int n = 0; n < channels.size(); n++) {
        // pour chaque x de gauche à droite
        for (int y = 0; y < channels[n].cols - 1; y++) {
            // pour chaque y de haut en bas
            for (int x = 1; x < channels[n].rows - 1; x++) {
                float ancien_pixel = channels[n].at<float>(x, y);
                float nouveau_pixel = (ancien_pixel > 128.0) ? 255.0 : 0.0;
                channels[n].at<float>(x, y) = nouveau_pixel;
                float erreur_quantification = ancien_pixel - nouveau_pixel;
                channels[n].at<float>(x + 1, y) =
                        channels[n].at<float>(x + 1, y) + (7.0 / 16.0 * erreur_quantification);
                channels[n].at<float>(x - 1, y + 1) =
                        channels[n].at<float>(x - 1, y + 1) + (3.0 / 16.0 * erreur_quantification);
                channels[n].at<float>(x, y + 1) =
                        channels[n].at<float>(x, y + 1) + (5.0 / 16.0 * erreur_quantification);
                channels[n].at<float>(x + 1, y + 1) =
                        channels[n].at<float>(x + 1, y + 1) + (1.0 / 16.0 * erreur_quantification);
            }
        }
    }

    merge(channels, output);
    output.convertTo(output, CV_8UC3);
}

/** FONCTION POUR LE TRAMAGE GENERIQUE **/
/* distance entre deux couleurs */
float distance_color_l2(Vec3f bgr1, Vec3f bgr2) {
    return sqrt(
            (bgr1[0] - bgr2[0]) * (bgr1[0] - bgr2[0])
            + (bgr1[1] - bgr2[1]) * (bgr1[1] - bgr2[1])
            + (bgr1[2] - bgr2[2]) * (bgr1[2] - bgr2[2]));
}

/* retourne l'indice de la couleur la plus proche du vecteur donné */
int best_color(Vec3f bgr, std::vector<Vec3f> colors) {
    int i = -1;
    float minDist = INFINITY;

    for (int n = 0; n < colors.size(); n++) {
        float dist = distance_color_l2(bgr, colors[n]);
        if (dist < minDist) {
            i = n;
            minDist = dist;
        }
    }

    return i;
}

/* retourne le vecteur d'erreur entre 2 couleurs */
Vec3f error_color(Vec3f bgr1, Vec3f bgr2) {
    return {bgr1[0] - bgr2[0], bgr1[1] - bgr2[1], bgr1[2] - bgr2[2]};
}

/* Tramage Floyd Steinberg générique */
Mat tramage_floyd_steinberg_generic(Mat input, std::vector<Vec3f> colors) {
    // Conversion de input en une matrice de 3 canaux flottants
    Mat fs;
    input.convertTo(fs, CV_32FC3, 1 / 255.0);

    // Algorithme Floyd-Steinberg
    for (int y = 0; y < fs.cols - 1; y++) {
        for (int x = 1; x < fs.rows - 1; x++) {
            Vec3f c = fs.at<Vec3f>(x, y);
            int i = best_color(c, colors);
            Vec3f e = error_color(c, colors[i]);
            fs.at<Vec3f>(x, y) = colors[i];

            // On propage e aux pixels voisins
            fs.at<Vec3f>(x + 1, y) = fs.at<Vec3f>(x + 1, y) + (7.0 / 16.0 * e);
            fs.at<Vec3f>(x - 1, y + 1) =
                    fs.at<Vec3f>(x - 1, y + 1) + (3.0 / 16.0 * e);
            fs.at<Vec3f>(x, y + 1) =
                    fs.at<Vec3f>(x, y + 1) + (5.0 / 16.0 * e);
            fs.at<Vec3f>(x + 1, y + 1) =
                    fs.at<Vec3f>(x + 1, y + 1) + (1.0 / 16.0 * e);
        }
    }

    // On reconvertit la matrice de 3 canaux flottants en BGR
    Mat output;
    fs.convertTo(output, CV_8UC3, 255.0);
    return output;
}

/** MAIN **/
int main(int, char *argv[]) {
    if (argv[1] == nullptr || argv[2] == nullptr) {
        std::cout << "\nUsage : ./main_color_img <nom-fichier-image> <egal | tram | genBGR | genCMYK | none>\n"
                  << std::endl;
        exit(1);
    }

    String filename = argv[1];
    String path = "/home/leodie/Documents/elodie/projects/TP1/";

    int old_value = 0;
    int value = 128;

    namedWindow("TP1 Color IMG");               // crée une fenêtre
    createTrackbar("track color", "TP1 Color IMG", &value, 255, nullptr); // un slider

    Mat f = imread(path + filename);        // lit l'image  donné en argument

    /** --- DEBUT DES APPELS DE FONCTIONS --- **/
    String functionToExecute = argv[2];

    if (functionToExecute == "none") {
        imshow("TP1 Color IMG", f);                // l'affiche dans la fenêtre
    } else if (functionToExecute == "egal") {
        /* Conversion BGR to HSV */
        cvtColor(f, f, COLOR_BGR2HSV);

        /* --- Histogrammes --- */
        std::vector<double> hist = histogramme(f);
        std::vector<double> histCumule = histogramme_cumule(hist);

        /* --- Egalisation --- */
        Mat equalizedImg = equalization(f, hist, histCumule);

        /* Conversion HSV to BGR */
        cvtColor(equalizedImg, equalizedImg, COLOR_HSV2BGR);
        imshow("TP1 Color IMG", equalizedImg);                // l'affiche dans la fenêtre

        Mat displayHistogrammes = afficheHistogrammes(hist, histCumule);
        namedWindow("Histogrammes Color IMG");
        imshow("Histogrammes Color IMG", displayHistogrammes);                // l'affiche dans la fenêtre
    } else if (functionToExecute == "tram") {
        /* --- Tramage Floyd Steinberg --- */
        Mat tramedImg(f.rows, f.cols, CV_32FC3);
        tramage_floyd_steinberg(f, tramedImg);
        imshow("TP1 Color IMG", tramedImg);                // l'affiche dans la fenêtre
    } else if (functionToExecute == "genBGR") {
        /* --- Tramage Floyd Steinberg Générique BGR --- */
        Vec3f blue({1.0, 0.0, 0.0});
        Vec3f green({0.0, 1.0, 0.0});
        Vec3f red({0.0, 0.0, 1.0});
        Vec3f black({0.0, 0.0, 0.0});
        Vec3f white({1.0, 1.0, 1.0});

        // Fonction générique avec les couleurs BGR
        std::vector<Vec3f> colorsBGR = {blue, green, red, black, white};
        Mat tramedImage = tramage_floyd_steinberg_generic(f, colorsBGR);
        imshow("TP1 Color IMG", tramedImage);                // l'affiche dans la fenêtre
    } else if (functionToExecute == "genCMYK") {
        /* --- Tramage Floyd Steinberg Générique CMYK --- */
        Vec3f cyan({1.0, 1.0, 0.0});
        Vec3f magenta({1.0, 0.0, 1.0});
        Vec3f yellow({0.0, 1.0, 1.0});
        Vec3f black({0.0, 0.0, 0.0});
        Vec3f white({1.0, 1.0, 1.0});

        // Fonction générique avec les couleurs CMYK
        std::vector<Vec3f> colorsCMJN = {cyan, magenta, yellow, black, white};
        Mat tramedImage = tramage_floyd_steinberg_generic(f, colorsCMJN);

        imshow("TP1 Color IMG", tramedImage);                // l'affiche dans la fenêtre
    } else {
        std::cout << "\nUsage : ./main_color_img <nom-fichier-image> <egal | tram | genBGR | genCMYK | none>\n"
                  << std::endl;
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