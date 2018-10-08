/*******************************************************/
/*                                                     */
/* SERGIO GONZALEZ VELAZQUEZ y DAVID CARNEROS PRADO    */
/*                                                     */
/*******************************************************/


#include <QApplication>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsPixmapItem>
#include <stdio.h>
#include <omp.h>

#define COLOUR_DEPTH 4

double computeGraySequential(QImage *image) {
  double start_time = omp_get_wtime();
  uchar *pixelPtr = image->bits();

  for (int ii = 0; ii < image->byteCount(); ii += COLOUR_DEPTH) {

    QRgb* rgbpixel = reinterpret_cast<QRgb*>(pixelPtr + ii);
    int gray = qGray(*rgbpixel);
    *rgbpixel = QColor(gray, gray, gray).rgba();
  }
  return omp_get_wtime() - start_time;
}

double computeGrayParallel(QImage *image) {
  double start_time = omp_get_wtime();
  uchar *pixelPtr = image->bits();

#pragma omp parallel for
  for (int ii = 0; ii < image->byteCount(); ii += COLOUR_DEPTH) {

    QRgb* rgbpixel = reinterpret_cast<QRgb*>(pixelPtr + ii);
    int gray = qGray(*rgbpixel);
    *rgbpixel = QColor(gray, gray, gray).rgba();
  }
  return omp_get_wtime() - start_time;
}

double computeGrayScanline(QImage *image) {
  double start_time = omp_get_wtime();
  int alto = image->height(); int ancho = image->width();
  int jj, gray; uchar* scan; QRgb* rgbpixel;
  for (int ii = 0; ii < alto; ii++) {

    scan = image->scanLine(ii);
    for (jj = 0; jj < ancho; jj++) {

      rgbpixel = reinterpret_cast<QRgb*>(scan + jj * COLOUR_DEPTH);
      gray = qGray(*rgbpixel);
      *rgbpixel = QColor(gray, gray, gray).rgba();
    }
  }
  return omp_get_wtime() - start_time;
}

/************ FUNCION AÑADIDA ***************/

/* Hemos paralelizado el primer "for", ya que es de reparto de trabajo.
El segundo "for" sin embargo, no es de reparto de trabajo, cuando entra una hebra
lo ejecuta secuencialmente de manera completa. Para ello hemos utilizado la directiva
pragma omp parallel for, y hemos declarado como variables privadas las que se utilizan
dentro del segundo bucle (jj,rgbpixel,gray y scan).
Para proteger como seccion critica la instrucion "scan = image->scanLine(ii);" Hemos
utlizado la directiva critical ya que no podiamos utilizar ni atomic ni reduction. */

double computeGrayScanlineParallel(QImage *image) {
  double start_time = omp_get_wtime();
  int alto = image->height(); int ancho = image->width();
  int jj, gray; uchar* scan; QRgb* rgbpixel;

  #pragma omp parallel for private(jj,rgbpixel,gray,scan)
  for (int ii = 0; ii < alto; ii++) {

    #pragma omp critical
    {
    scan = image->scanLine(ii);
    }

    for (jj = 0; jj < ancho; jj++) {

      rgbpixel = reinterpret_cast<QRgb*>(scan + jj * COLOUR_DEPTH);
      gray = qGray(*rgbpixel);
      *rgbpixel = QColor(gray, gray, gray).rgba();
    }
  }
  return omp_get_wtime() - start_time;
}


int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QGraphicsScene scene;
    QGraphicsView view(&scene);
    QPixmap qp = QPixmap("test_1080p.bmp"); // ("c:\\test_1080p.bmp");
    if(qp.isNull())
    {
        printf("image not found\n");
		return -1;
    }

    QImage image = qp.toImage();

    QImage seqImage(image);
    double computeTime = computeGraySequential(&seqImage);
    printf("sequential time: %0.9f seconds\n", computeTime);

    QImage auxImage(image);
    computeTime = computeGrayParallel(&auxImage);
    printf("parallel time: %0.9f seconds\n", computeTime);

	if (auxImage == seqImage) printf("Algoritmo secuencial y paralelo dan la misma imagen\n");
	else printf("Algoritmo secuencial y paralelo dan distinta imagen\n");

    auxImage = image;
    computeTime = computeGrayScanline(&auxImage);
    printf("scanline time: %0.9f seconds\n", computeTime);


	if (auxImage == seqImage) printf("Algoritmo básico y 'scanline' dan la misma imagen\n");
	else printf("Algoritmo básico y 'scanline' dan distinta imagen\n");

  /* Generamos la imagen en escala de grises con la funcion paralelizado  y
  mostramos su tiempo */
  auxImage = image;
  computeTime = computeGrayScanlineParallel(&auxImage);
  printf("scanline Parallel time: %0.9f seconds\n",computeTime);


  if (auxImage == seqImage) printf("Algoritmo scanline secuencial y scanline parallel dan la misma imagen\n");
	else printf("Algoritmo scanline secuencial y scanline parallel dan distinta imagen\n");

    QPixmap pixmap = pixmap.fromImage(auxImage);
    QGraphicsPixmapItem *item = new QGraphicsPixmapItem(pixmap);
    scene.addItem(item);

    view.show();
    return a.exec();
}
