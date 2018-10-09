/*************************************************************
*
* 			SERGIO GONZALEZ VELAZQUEZ y DAVID CARNEROS PRADO
*
*
*
*
*
*
*
*
*
*
*
*
*
*
*
************************************************************/


#include <QApplication>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsPixmapItem>
#include <stdio.h>
#include <omp.h>

#define COLOUR_DEPTH 4

omp_lock_t lock[256];


/* -------- VERSION SECUENCIAL -------- */

double computeHistogramSecuencial(QImage *image,int histgr[]) {
  double start_time = omp_get_wtime();
  uchar *pixelPtr = image->bits();

  for (int ii = 0; ii < image->byteCount(); ii += COLOUR_DEPTH) {

    QRgb* rgbpixel = reinterpret_cast<QRgb*>(pixelPtr + ii);
    int gray = qGray(*rgbpixel);
    histgr[gray]++;
  }
  return omp_get_wtime() - start_time;
}


/* -------- VERSION PARALELA CON CRITICAL -------- */

double computeHistogramCritical(QImage *image,int histgr[]) {
  double start_time = omp_get_wtime();
  uchar *pixelPtr = image->bits();

  #pragma omp parallel for 

  for (int ii = 0; ii < image->byteCount(); ii += COLOUR_DEPTH) {

    QRgb* rgbpixel = reinterpret_cast<QRgb*>(pixelPtr + ii);
    int gray = qGray(*rgbpixel);

   #pragma omp critical
    histgr[gray]++;
  }

  return omp_get_wtime() - start_time;
}


/* -------- VERSION PARALELA CON ATOMIC -------- */



double computeHistogramAtomic(QImage *image,int histgr[]) {
  double start_time = omp_get_wtime();
  uchar *pixelPtr = image->bits();

  #pragma omp parallel for 

  for (int ii = 0; ii < image->byteCount(); ii += COLOUR_DEPTH) {

    QRgb* rgbpixel = reinterpret_cast<QRgb*>(pixelPtr + ii);
    int gray = qGray(*rgbpixel);

   #pragma omp atomic
    histgr[gray]++;
  }

  return omp_get_wtime() - start_time;
}

/* -------- VERSION PARALELA CON REDUCTION-------- */
double computeHistogramReduction(QImage *image,int histgr[]) {
  double start_time = omp_get_wtime();
  uchar *pixelPtr = image->bits();

  #pragma omp parallel for reduction(+:histgr[:256])

  for (int ii = 0; ii < image->byteCount(); ii += COLOUR_DEPTH) {

    QRgb* rgbpixel = reinterpret_cast<QRgb*>(pixelPtr + ii);
    int gray = qGray(*rgbpixel);

    histgr[gray]++;
  }

  return omp_get_wtime() - start_time;
}



/* -------- VERSION PARALELA CON LOCKS DE BAJO NIVEL------- */
void createLocks(){
	for(int i=0;i<256;i++){
		omp_init_lock(&lock[i]);
	}
}

void destroyLocks(){
	for(int i=0;i<256;i++){
		omp_destroy_lock(&lock[i]);
	}
}

double computeHistogramLock(QImage *image,int histgr[]) {

  double start_time = omp_get_wtime();
  uchar *pixelPtr = image->bits();
  createLocks();

  #pragma omp parallel for 

  for (int ii = 0; ii < image->byteCount(); ii += COLOUR_DEPTH) {

    QRgb* rgbpixel = reinterpret_cast<QRgb*>(pixelPtr + ii);
    int gray = qGray(*rgbpixel);

    omp_set_lock(&lock[gray]); 		// Entrada a la seccion critica
    histgr[gray]++;					// Seccion critica
    omp_unset_lock(&lock[gray]); 	// Salida de la seccion critica
  }

  destroyLocks();		//Destruimos los cerrojos para liberar memoria

  return omp_get_wtime() - start_time;
}


bool compararHist(int histgr[],int histgrAux[]){

	bool igual = true;

	for (int i=0;i<256 && igual;i++){
		igual = (histgr[i]==histgrAux[i]);
	}

	return igual;
}


int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QGraphicsScene scene;
    QGraphicsView view(&scene);
    QPixmap qp = QPixmap("test_1080p.bmp"); // ("c:\\test_1080p.bmp");

    int histgr[256];
    int histgrAux[256];

    memset(histgr,0,sizeof(histgr));		//Se inicializan los componentes a 0
    memset(histgrAux,0,sizeof(histgrAux));

    if(qp.isNull())
    {
        printf("image not found\n");
		return -1;
    }

    QImage image = qp.toImage();

    QImage seqImage(image);

    double computeTime = computeHistogramSecuencial(&image,histgr);
    printf("sequential time: %0.9f seconds\n", computeTime);

    computeTime = computeHistogramCritical(&image,histgrAux);
    printf("critical time: %0.9f seconds\n", computeTime);

    if (compararHist(histgr,histgrAux)) printf("Algoritmo secuencial y paralelo con 'critical' dan el mismo datagrama \n");
    else printf("Algoritmo secuencial y paralelo con 'critical' NO dan el mismo datagrama \n");

	memset(histgrAux,0,sizeof(histgrAux));
    computeTime = computeHistogramAtomic(&image,histgrAux);
    printf("Atomic time: %0.9f seconds\n", computeTime);
    if (compararHist(histgr,histgrAux)) printf("Algoritmo secuencial y paralelo con 'atomic' dan el mismo datagrama \n");
    else printf("Algoritmo secuencial y paralelo con 'atomic' NO dan el mismo datagrama \n");


    
	memset(histgrAux,0,sizeof(histgrAux));
    computeTime = computeHistogramReduction(&image,histgrAux);
    printf("Reduction time: %0.9f seconds\n", computeTime);
    if (compararHist(histgr,histgrAux)) printf("Algoritmo secuencial y paralelo con 'reduction' dan el mismo datagrama \n");
    else printf("Algoritmo secuencial y paralelo con 'reduction' NO dan el mismo datagrama \n");

	
	memset(histgrAux,0,sizeof(histgrAux));
    computeTime = computeHistogramLock(&image,histgrAux);
    printf("Locks time: %0.9f seconds\n", computeTime);
    if (compararHist(histgr,histgrAux)) printf("Algoritmo secuencial y paralelo con 'locks' dan el mismo datagrama \n");
    else printf("Algoritmo secuencial y paralelo con 'locks' NO dan el mismo datagrama \n");


   /*
   QPixmap pixmap = pixmap.fromImage(auxImage);
   QGraphicsPixmapItem *item = new QGraphicsPixmapItem(pixmap);
   scene.addItem(item);

   view.show();



    return a.exec();
    */
}
