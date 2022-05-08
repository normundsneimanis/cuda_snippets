/*
 * Harris corners detector algorithm on GPU
 * Arnis Lektauers 2019 Riga Technical University
 */
#include "bitmap_image.h"

#include <iostream>
#include <stdlib.h>
#include <time.h>

#include <stdio.h>
#include <string.h>
#include <math.h>

const RGBA RGBA::Red(255, 0, 0);
const RGBA RGBA::Green(0, 255, 0);
const RGBA RGBA::Blue(0, 0, 255);
const RGBA RGBA::Black(0, 0, 0);

unsigned int RGBAToInt(const RGBA& rgba) {
	return ((unsigned int)(rgba.red) << 24) |
           ((unsigned int)(rgba.green) << 16) |
           ((unsigned int)(rgba.blue) <<  8) |
           ((unsigned int)(rgba.alpha));
}

bool operator != (const RGBA& l, const RGBA& r) {
	return RGBAToInt(l) != RGBAToInt(r);
}

bool operator < (const RGBA& l, const RGBA& r ) { 
    return RGBAToInt(l) < RGBAToInt(r);
}

inline WORD flipWORD(WORD in) { 
	return ((in >> 8) | (in << 8)); 
}

inline DWORD flipDWORD(DWORD in) {
	return (((in & 0xFF000000) >> 24) | ((in & 0x000000FF) << 24) | ((in & 0x00FF0000) >> 8) | ((in & 0x0000FF00) << 8));
}

inline int intSquare(int number) { 
	return number * number; 
}

int intPow( int base, int exponent) {
	int output = 1;
	for (int i = 0; i < exponent; i++) { 
		output *= base; 
	}
	return output;
}

struct BitmapFileHeader {
	WORD bfType;
	DWORD bfSize;
	WORD bfReserved1;
	WORD bfReserved2;
	DWORD bfOffBits; 

	BitmapFileHeader() : bfType(19778), bfReserved1(0), bfReserved2(0) {}

	void switchEndianess(void) {
		bfType = flipWORD(bfType);
		bfSize = flipDWORD(bfSize);
		bfReserved1 = flipWORD(bfReserved1);
		bfReserved2 = flipWORD(bfReserved2);
		bfOffBits = flipDWORD(bfOffBits);
	}
};

struct DIBHeader {
	DWORD biSize;
	DWORD biWidth;
	DWORD biHeight;
	WORD  biPlanes;
	WORD  biBitCount;
	DWORD biCompression;
	DWORD biSizeImage;
	DWORD biXPelsPerMeter;
	DWORD biYPelsPerMeter;
	DWORD biClrUsed;
	DWORD biClrImportant;

	DIBHeader() : biPlanes(1), biCompression(0), biClrUsed(0), biClrImportant(0) {} 

	void switchEndianess(void) {
		biSize = flipDWORD(biSize);
		biWidth = flipDWORD(biWidth);
		biHeight = flipDWORD(biHeight);
		biPlanes = flipWORD(biPlanes);
		biBitCount = flipWORD(biBitCount);
		biCompression = flipDWORD(biCompression);
		biSizeImage = flipDWORD(biSizeImage);
		biXPelsPerMeter = flipDWORD(biXPelsPerMeter);
		biYPelsPerMeter = flipDWORD(biYPelsPerMeter);
		biClrUsed = flipDWORD(biClrUsed);
		biClrImportant = flipDWORD(biClrImportant);
		return;
	}
};

bool safeFileRead(char* buffer, int size, int number, FILE* fp) {
	using namespace std;
	int itemsRead;
	if (feof(fp)) { 
		return false; 
	}
	itemsRead = (int)fread(buffer , size , number, fp);
	if (itemsRead < number) { 
		return false; 
	}
	return true;
}

inline bool isBigEndian() {
	short word = 0x0001;
	if ((*(char*)&word) != 0x01) { 
		return true; 
	}
	return false;
}

BitmapImage::~BitmapImage() {
	if (_rawData) {
		delete [] _rawData;
	}

	if (_paletteColors) {
		delete [] _paletteColors;
	}
}

BitmapImage::BitmapImage(const BitmapImage& input) : 
    _rawData(NULL), _paletteColors(NULL), _dataSize(input._dataSize), 
	_bitDepth(input._bitDepth), _stride(input._stride), _grayScale(input._grayScale) {
	
	setSize(input.getWidth(), input.getHeight());

	if (input._paletteColors) {
		int colorCount = input.getNumberOfColors();

		_paletteColors = new RGBA [colorCount]; 
		memcpy(_paletteColors, input._paletteColors, colorCount * sizeof(RGBA)); 
	}
}

void BitmapImage::copy32bpp(const BitmapImage& source) {
	if (_bitDepth == 8) {
		if (isGrayScale()) {
			_dataSize = _width * _height * sizeof(RGBA); 
			BYTE* newRawData = new BYTE[_dataSize]; 

			for (int y = 0; y < _height; y++) {
				for (int x = 0; x < _width; x++) {
					int offset = x + y * _width;
					BYTE v = source._rawData[offset];
					((RGBA*)&newRawData[4 * offset])->setGrayScale(v);
				}
			}

			if (_rawData) {
				delete [] _rawData;
			}

			_rawData = newRawData;
		}
				
		_bitDepth = 32;
		calculateStride();
		
		if (_paletteColors) {
			delete [] _paletteColors;
			_paletteColors = NULL;
		}
	}
}

bool BitmapImage::loadFromFile(const std::string& fileName) {
	using namespace std;

	FILE* fp = fopen(fileName.c_str(), "rb" );

	if (fp == NULL) {
		cout << "Bitmap error: Cannot open file " << fileName << " for input." << endl;
		return false;
	}

	BitmapFileHeader fileHeader;

	bool notCorrupted = safeFileRead((char*)&(fileHeader.bfType) , sizeof(WORD), 1, fp);

	bool isBitmap = false;
	if (isBigEndian() && fileHeader.bfType == 16973 ) { 
		isBitmap = true; 
	}
	if (!isBigEndian() && fileHeader.bfType == 19778) { 
		isBitmap = true; 
	}

	if (!isBitmap) {
		cerr << "Bitmap error: " << fileName << " is not a Windows BMP file!" << endl; 
		fclose(fp); 
		return false;
	}

	notCorrupted &= safeFileRead((char*) &(fileHeader.bfSize), sizeof(DWORD), 1, fp); 
	notCorrupted &= safeFileRead((char*) &(fileHeader.bfReserved1), sizeof(WORD), 1, fp);
	notCorrupted &= safeFileRead((char*) &(fileHeader.bfReserved2), sizeof(WORD), 1, fp);
	notCorrupted &= safeFileRead((char*) &(fileHeader.bfOffBits), sizeof(DWORD), 1, fp);	

	if (isBigEndian()) { 
		fileHeader.switchEndianess(); 
	}

	DIBHeader dbiHeader; 

	notCorrupted &= safeFileRead((char*) &(dbiHeader.biSize), sizeof(DWORD), 1, fp);
	notCorrupted &= safeFileRead((char*) &(dbiHeader.biWidth), sizeof(DWORD), 1, fp); 
	notCorrupted &= safeFileRead((char*) &(dbiHeader.biHeight), sizeof(DWORD), 1, fp);
	notCorrupted &= safeFileRead((char*) &(dbiHeader.biPlanes), sizeof(WORD), 1, fp); 
	notCorrupted &= safeFileRead((char*) &(dbiHeader.biBitCount), sizeof(WORD), 1, fp);

	notCorrupted &= safeFileRead((char*) &(dbiHeader.biCompression), sizeof(DWORD), 1, fp);
	notCorrupted &= safeFileRead((char*) &(dbiHeader.biSizeImage), sizeof(DWORD), 1, fp);
	notCorrupted &= safeFileRead((char*) &(dbiHeader.biXPelsPerMeter), sizeof(DWORD), 1, fp);
	notCorrupted &= safeFileRead((char*) &(dbiHeader.biYPelsPerMeter), sizeof(DWORD), 1, fp);
	notCorrupted &= safeFileRead((char*) &(dbiHeader.biClrUsed), sizeof(DWORD), 1, fp);
	notCorrupted &= safeFileRead((char*) &(dbiHeader.biClrImportant), sizeof(DWORD), 1, fp);

	if (isBigEndian()) { 
		dbiHeader.switchEndianess(); 
	}

	if (!notCorrupted) {
		cout << "EasyBMP Error: " << fileName << " is obviously corrupted." << endl;
		fclose(fp);
		return false;
	} 

	// the file is RLE compressed
	if (dbiHeader.biCompression == 1 || dbiHeader.biCompression == 2) {
		cout << "Error: " << fileName << " is (RLE) compressed." << endl << " Not supported compression." << endl;
		fclose(fp);
		return false; 
	}

	// something strange is going on it's probably an OS2 bitmap file
	if (dbiHeader.biCompression > 3) {
		cout << "Error: " << fileName << " is in an unsupported format." << endl << " (biCompression = "  << dbiHeader.biCompression << ")" << endl
				<< "               The file is probably an old OS2 bitmap or corrupted." << endl;
		fclose(fp);
		return false; 
	}

	if (dbiHeader.biCompression == 3 && dbiHeader.biBitCount != 16) {
		cout << "Error: " << fileName << " uses bit fields and is not a" << endl
				<< "               16-bit file. This is not supported." << endl;
		fclose(fp);
		return false; 
	}

	int tempBitDepth = (int)dbiHeader.biBitCount;
	if (tempBitDepth != 1  && tempBitDepth != 4 && tempBitDepth != 8  && tempBitDepth != 16 && tempBitDepth != 24 && tempBitDepth != 32) {
		cout << "Error: " << fileName << " has unrecognized bit depth." << endl;
		fclose(fp);
		return false;
	}
	setBitDepth((int)dbiHeader.biBitCount); 

	if ((int)dbiHeader.biWidth <= 0 || (int)dbiHeader.biHeight <= 0)  {
		cout << "Error: " << fileName << " has a non-positive width or height." << endl;
		fclose(fp);
		return false;
	} 
	/*double bytesPerPixel = ((double)_bitDepth) / 8.0;
		
	double bytesPerRow = bytesPerPixel * (double)dbiHeader.biWidth;
	bytesPerRow = ceil(bytesPerRow);

	int bytePaddingPerRow = 4 - ((int)(bytesPerRow)) % 4;
	if (bytePaddingPerRow == 4) { 
		bytePaddingPerRow = 0; 
	}*/

	// skip blank data if bfOffBits so indicates
	int bytesToSkip = fileHeader.bfOffBits - 54;
	if (_bitDepth < 16) { 
		bytesToSkip -= 4 * intPow(2, _bitDepth); 
	}
	if (_bitDepth == 16 && dbiHeader.biCompression == 3) { 
		bytesToSkip -= 3 * 4; 
	}
	if (bytesToSkip < 0) { 
		bytesToSkip = 0; 
	}
	if (bytesToSkip > 0 && _bitDepth != 16) {
		cout << "Warning: Extra meta data detected in file. Data will be skipped. " << endl;

		BYTE* tempSkipBYTE = new BYTE[bytesToSkip];
		safeFileRead((char*)tempSkipBYTE, bytesToSkip, 1, fp);   
		delete [] tempSkipBYTE;
	} 
	
	// read the palette
	if (_bitDepth < 16) {
		// determine the number of colors specified in the 
		// color table
		int numberOfColorsToRead = ((int)fileHeader.bfOffBits - 54) / 4;  
		if (numberOfColorsToRead > intPow(2, _bitDepth)) { 
			numberOfColorsToRead = intPow(2, _bitDepth); 
		}

		if (numberOfColorsToRead < getNumberOfColors()) {
			cout << "Warning: file " << fileName << " has an underspecified" << endl
			 	 << "                 color table. The table will be padded with extra" << endl
				 << "                 white (255,255,255,0) entries." << endl;
		}

		int n;
		for (n = 0; n < numberOfColorsToRead; n++) {
			safeFileRead((char*) &(_paletteColors[n]), 4, 1, fp);     
		}

		const RGBA WHITE(255, 255, 255, 0); 
		for (n = numberOfColorsToRead; n < getNumberOfColors(); n++) {			
			_paletteColors[n] = WHITE;
		}
	}

	setSize((int)dbiHeader.biWidth, (int)dbiHeader.biHeight);
	calculateStride();
	
	// This code reads 1, 4, 8, 24, and 32-bpp files 
	// with a more-efficient buffered technique.

	bool grayScale = isGrayScale();

	int i,j;
	if (_bitDepth != 16) {
		int bufferSize = _width * _bitDepth / 8;
		while (8 * bufferSize < _width * _bitDepth) { 
			bufferSize++; 
		}
		while (bufferSize % 4) { 
			bufferSize++; 
		}
		
		BYTE* buffer = new BYTE [bufferSize];
		j = _height - 1;
		while (j > -1 ) {
			int bytesRead = (int) fread((char*)buffer, 1, bufferSize, fp);
			if (bytesRead < bufferSize) {
				j = -1; 
				cout << "Error: Could not read proper amount of data." << endl;
			} else {
				bool success = false;
				/*if (_bitDepth == 1) { 
					success = read1bitRow(buffer, bufferSize, j); 
				}
				if (_bitDepth == 4) { 
					success = read4bitRow(buffer, bufferSize, j); 
				}*/
				if (_bitDepth == 8) { 
					success = read8bitRow(buffer, bufferSize, j, grayScale); 
				}
				if (_bitDepth == 24) { 
					success = read24bitRow(buffer, bufferSize, j);
				}
				/*if (_bitDepth == 32) { 
					success = read32bitRow(buffer, bufferSize, j); 
				}*/
				if (!success) {
					cout << "Error: Could not read enough pixel data!" << endl;
					j = -1;
				}
			}   
			j--;
		}
		delete [] buffer; 
	}

	fclose(fp);
	return true;
}

void BitmapImage::calculateStride() {
	double bytesPerPixel = _bitDepth / 8.0;
	double bytesPerRow = bytesPerPixel * _width;
	bytesPerRow = ceil(bytesPerRow);

	int bytePaddingPerRow = 4 - ((int)bytesPerRow) % 4;
	if (bytePaddingPerRow == 4) { 
		bytePaddingPerRow = 0; 
	} 

	_stride = (int)bytesPerRow + bytePaddingPerRow;
}

bool BitmapImage::saveToFile(const std::string& fileName) {
	using namespace std;
	
	FILE* fp = fopen(fileName.c_str(), "wb");

	if (fp == NULL) {
		cout << "Bitmap error: Cannot open file " << fileName << " for output." << endl;
		return false;
	}

	double totalPixelBytes = _height * _stride;

	double paletteSize = 0;
	if (_bitDepth == 1 || _bitDepth == 4 || _bitDepth == 8) { 
		paletteSize = intPow(2, _bitDepth) * 4.0; 
	}

	// leave some room for 16-bit masks 
	if (_bitDepth == 16) { 
		paletteSize = 3 * 4; 
	}

	double totalFileSize = 14 + 40 + paletteSize + totalPixelBytes;

	// write the file header 
	BitmapFileHeader fileHeader;
	fileHeader.bfSize = (DWORD)totalFileSize; 
	fileHeader.bfReserved1 = 0; 
	fileHeader.bfReserved2 = 0; 
	fileHeader.bfOffBits = (DWORD)(14 + 40 + paletteSize);  

	if (isBigEndian()) { 
		fileHeader.switchEndianess(); 
	}

	fwrite((char*) &(fileHeader.bfType), sizeof(WORD), 1, fp);
	fwrite((char*) &(fileHeader.bfSize), sizeof(DWORD), 1, fp);
	fwrite((char*) &(fileHeader.bfReserved1), sizeof(WORD), 1, fp);
	fwrite((char*) &(fileHeader.bfReserved2), sizeof(WORD), 1, fp);
	fwrite((char*) &(fileHeader.bfOffBits), sizeof(DWORD), 1, fp);

	// write the info header 
	DIBHeader dibHeader;
	dibHeader.biSize = 40;
	dibHeader.biWidth = _width;
	dibHeader.biHeight = _height;
	dibHeader.biPlanes = 1;
	dibHeader.biBitCount = _bitDepth;
	dibHeader.biCompression = 0;
	dibHeader.biSizeImage = (DWORD)totalPixelBytes;
	/*if (_xPelsPerMeter) { 
		dibHeader.biXPelsPerMeter = _xPelsPerMeter; 
	} else {*/ 
		dibHeader.biXPelsPerMeter = DEF_X_PELS_PER_METER; 
	//}
	/*if (_yPelsPerMeter) { 
		dibHeader.biYPelsPerMeter = _yPelsPerMeter; 
	} else {*/ 
		dibHeader.biYPelsPerMeter = DEF_Y_PELS_PER_METER; 
	//}

	dibHeader.biClrUsed = 0;
	dibHeader.biClrImportant = 0;

	// indicates that we'll be using bit fields for 16-bit files
	if (_bitDepth == 16 ) { 
		dibHeader.biCompression = 3; 
	}

	if (isBigEndian()) { 
		fileHeader.switchEndianess(); 
	}

	fwrite( (char*) &(dibHeader.biSize), sizeof(DWORD), 1, fp);
	fwrite( (char*) &(dibHeader.biWidth), sizeof(DWORD), 1, fp);
	fwrite( (char*) &(dibHeader.biHeight), sizeof(DWORD), 1, fp);
	fwrite( (char*) &(dibHeader.biPlanes), sizeof(WORD), 1, fp);
	fwrite( (char*) &(dibHeader.biBitCount), sizeof(WORD), 1, fp);
	fwrite( (char*) &(dibHeader.biCompression), sizeof(DWORD), 1, fp);
	fwrite( (char*) &(dibHeader.biSizeImage), sizeof(DWORD), 1, fp);
	fwrite( (char*) &(dibHeader.biXPelsPerMeter), sizeof(DWORD), 1, fp);
	fwrite( (char*) &(dibHeader.biYPelsPerMeter), sizeof(DWORD), 1, fp); 
	fwrite( (char*) &(dibHeader.biClrUsed), sizeof(DWORD), 1, fp);
	fwrite( (char*) &(dibHeader.biClrImportant), sizeof(DWORD), 1, fp);

	// write the palette 
	if (_bitDepth == 1 || _bitDepth == 4 || _bitDepth == 8) {
		int numberOfColors = intPow(2, _bitDepth);

		// if there is no palette, create one 
		if (!_paletteColors) {
			_paletteColors = new RGBA[numberOfColors]; 
			createStandardColorTable(); 
		}

		int n;
		for (n = 0; n < numberOfColors; n++) { 
			fwrite((char*) &(_paletteColors[n]), 4, 1, fp); 
		}
	}

	// write the pixels 
	int i,j;
	if (_bitDepth != 16) {  
		BYTE* buffer;
		int bufferSize = (int)((_width * _bitDepth) / 8.0);
		while (8 * bufferSize < _width * _bitDepth) { 
			bufferSize++; 
		}
		while (bufferSize % 4) { 
			bufferSize++; 
		}

		buffer = new BYTE[bufferSize];
		for (j = 0; j < bufferSize; j++) { 
			buffer[j] = 0; 
		}

		j = _height - 1;

		while (j > -1) {
			bool success = false;
			if (_bitDepth == 32) { 
				success = write32bitRow(buffer, bufferSize, j); 
			} else if (_bitDepth == 24) { 
				success = write24bitRow(buffer, bufferSize, j); 
			} else if (_bitDepth == 8) { 
				success = write8bitRow(buffer, bufferSize, j); 
			}
			/*if (_bitDepth == 4) { 
				success = write4bitRow(buffer, bufferSize, j); 
			}
			if (_bitDepth == 1) { 
				success = write1bitRow(buffer, bufferSize, j);
			}*/

			if (success) {
				int bytesWritten = (int)fwrite((char*)buffer, 1, bufferSize, fp);
				if (bytesWritten != bufferSize) { 
					success = false; 
				}
			}
			if (!success) {
				cout << "Error: Could not write proper amount of data." << endl;
				j = -1; 
			}
			j--; 
		}

		delete [] buffer;
	}

	fclose(fp);

	cout << "Saved bitmap file '" << fileName << "'" << endl;

	return true;
}

bool BitmapImage::write8bitRow(BYTE* buffer, int bufferSize, int row) {
	if (_width > bufferSize) { 
		return false; 
	}

	if (isGrayScale()) {
		memcpy(buffer, &_rawData[row * _width], _width); 
	} else {
		for (int x = 0; x < _width; x++) { 
			buffer[x] = findClosestColor(*((RGBA*)&_rawData[x + row * _width])); 
		}
	}
	return true;
}

bool BitmapImage::write24bitRow(BYTE* buffer, int bufferSize, int row) { 
	if (_width * 3 > bufferSize) { 
		return false; 
	}
	for (int x = 0; x < _width; x++) { 
		memcpy(&buffer[3 * x], &_rawData[4 * (x + row * _width)], 3); 
	}
	return true;
}

bool BitmapImage::write32bitRow(BYTE* buffer, int bufferSize, int row) {
	if (_width * 4 > bufferSize) { 
		return false; 
	}
	memcpy(buffer, &_rawData[4 * row * _width], 4 * _width);
	return true;
}

BYTE BitmapImage::findClosestColor(const RGBA& input) {
	ColorTable::iterator result = _colorTable.find(input);
	if (result != _colorTable.end()) {
		return result->second;
	}

	using namespace std;

	int i = 0;
	int numberOfColors = getNumberOfColors();
	BYTE bestI = 0;
	int bestMatch = 999999;

	while (i < numberOfColors) {
		RGBA attempt = _paletteColors[i];
		
		int tempMatch = intSquare( (int) attempt.red - (int) input.red)
			+ intSquare((int) attempt.green - (int) input.green)
			+ intSquare((int) attempt.blue - (int) input.blue);
		
		if (tempMatch < bestMatch) { 
			bestI = (BYTE)i; 
			bestMatch = tempMatch; 
		}
		if (bestMatch < 1) { 
			i = numberOfColors; 
		}
		i++;
	}

	_colorTable[input] = bestI;
	return bestI;
}

bool BitmapImage::read24bitRow(BYTE* buffer, int bufferSize, int row) { 
	if (_width * 3 > bufferSize) { 
		return false; 
	}
	for (int x = 0; x < _width; x++) { 
		memcpy(&_rawData[4 * (x + row * _width)], &buffer[3 * x], 3); 
	}
	return true;
}

bool BitmapImage::read8bitRow(BYTE* buffer, int bufferSize, int row, bool grayScale) {
	if (_width > bufferSize) { 
		return false; 
	}

	int offset = row * _width;

	if (grayScale) {
		memcpy(&_rawData[offset], &buffer[0], _width);
		/*for (int x = 0; x < _width; x++) {
			int colorIdx = buffer[x];
			_rawData[x + offset] = _paletteColors[colorIdx].getGrayScale();
		}*/
	} else {
		for (int x = 0; x < _width; x++) {
			int colorIdx = buffer[x];
			memcpy(&_rawData[4 * (x + offset)], &_paletteColors[colorIdx], 4); 
		}
	}

	return true;
}

int BitmapImage::getNumberOfColors(void) const {
	if (_bitDepth == 32) { 
		return intPow(2, 24); 
	} else {
		return intPow(2, _bitDepth);
	}
}

void BitmapImage::setBitDepth(int newDepth) {
	_bitDepth = newDepth;
		
	if (_paletteColors) { 
		delete [] _paletteColors; 
	}

	int numberOfColors = intPow(2, _bitDepth);
	if (_bitDepth == 1 || _bitDepth == 4 || _bitDepth == 8) { 
		_paletteColors = new RGBA [numberOfColors]; 		
	} else { 
		_paletteColors = NULL; 
	} 
	
	if (_bitDepth == 1 || _bitDepth == 4 || _bitDepth == 8) { 
		createStandardColorTable(); 
	}
}

bool BitmapImage::createStandardColorTable(void) {
	using namespace std;
	if (_bitDepth != 1 && _bitDepth != 4 && _bitDepth != 8) {
		cout << "Warning: Attempted to create color table at a bit" << endl
			 << "                 depth that does not require a color table." << endl
			 << "                 Ignoring request." << endl;
		return false;
	}

	if (_bitDepth == 1) {
		for (int i = 0; i < 2 ; i++) {
			_paletteColors[i].red = i * 255;
			_paletteColors[i].green = i * 255;
			_paletteColors[i].blue = i * 255;
			_paletteColors[i].alpha = 0;
		} 
		return true;
	} 

	if (_bitDepth == 4 ) {
		int i = 0;
		int j, k, ell;

		// simplify the code for the first 8 colors
		for (ell = 0 ; ell < 2 ; ell++) {
			for (k = 0; k < 2 ; k++) {
				for (j = 0 ; j < 2; j++) {
					_paletteColors[i].red = j * 128; 
					_paletteColors[i].green = k * 128;
					_paletteColors[i].blue = ell * 128;
					i++;
				}
			}
		}

		// simplify the code for the last 8 colors
		for (ell = 0; ell < 2; ell++) {
			for (k = 0; k < 2; k++) {
				for (j = 0; j < 2; j++) {
					_paletteColors[i].red = j * 255;
					_paletteColors[i].green = k * 255; 
					_paletteColors[i].blue = ell * 255;
					i++;
				}
			}
		}

		// overwrite the duplicate color
		i = 8; 
		_paletteColors[i].red = 192;
		_paletteColors[i].green = 192;
		_paletteColors[i].blue = 192;

		for (i = 0; i < 16 ; i++) { 
			_paletteColors[i].alpha = 0; 
		}
		return true;
	}

	if (_bitDepth == 8) {
		int i = 0;
		int j, k, ell;

		// do an easy loop, which works for all but colors 
		// 0 to 9 and 246 to 255
		for (ell = 0; ell < 4; ell++)  {
			for (k = 0; k < 8; k++) {
				for (j = 0; j < 8 ; j++) {
					_paletteColors[i].red = j * 32; 
					_paletteColors[i].green = k * 32;
					_paletteColors[i].blue = ell * 64;
					_paletteColors[i].alpha = 0;
					i++;
				}
			}
		} 

		// now redo the first 8 colors  
		i = 0;
		for (ell = 0; ell < 2; ell++)  {
			for (k = 0; k < 2; k++) {
				for (j = 0; j < 2; j++) {
					_paletteColors[i].red = j * 128;
					_paletteColors[i].green = k * 128;
					_paletteColors[i].blue = ell * 128;
					i++;
				}
			}
		} 

		// overwrite colors 7, 8, 9
		i = 7;
		_paletteColors[i].red = 192;
		_paletteColors[i].green = 192;
		_paletteColors[i].blue = 192;
		i++; // 8
		_paletteColors[i].red = 192;
		_paletteColors[i].green = 220;
		_paletteColors[i].blue = 192;
		i++; // 9
		_paletteColors[i].red = 166;
		_paletteColors[i].green = 202;
		_paletteColors[i].blue = 240;

		// overwrite colors 246 to 255 
		i=246;
		_paletteColors[i].red = 255;
		_paletteColors[i].green = 251;
		_paletteColors[i].blue = 240;
		i++; // 247
		_paletteColors[i].red = 160;
		_paletteColors[i].green = 160;
		_paletteColors[i].blue = 164;
		i++; // 248
		_paletteColors[i].red = 128;
		_paletteColors[i].green = 128;
		_paletteColors[i].blue = 128;
		i++; // 249
		_paletteColors[i].red = 255;
		_paletteColors[i].green = 0;
		_paletteColors[i].blue = 0;
		i++; // 250
		_paletteColors[i].red = 0;
		_paletteColors[i].green = 255;
		_paletteColors[i].blue = 0;
		i++; // 251
		_paletteColors[i].red = 255;
		_paletteColors[i].green = 255;
		_paletteColors[i].blue = 0;
		i++; // 252
		_paletteColors[i].red = 0;
		_paletteColors[i].green = 0;
		_paletteColors[i].blue = 255;
		i++; // 253
		_paletteColors[i].red = 255;
		_paletteColors[i].green = 0;
		_paletteColors[i].blue = 255;
		i++; // 254
		_paletteColors[i].red = 0;
		_paletteColors[i].green = 255;
		_paletteColors[i].blue = 255;
		i++; // 255
		_paletteColors[i].red = 255;
		_paletteColors[i].green = 255;
		_paletteColors[i].blue = 255;

		return true;
	}
	return true;
}

bool BitmapImage::isGrayScale() const {
	if (_grayScale >= 0) {
		return _grayScale;
	}

	_grayScale = 0;
	if (!_paletteColors) {
		return false;
	}

	int colorCount = getNumberOfColors();

	for (int i = 0; i < colorCount; i++) {
		RGBA* color = &_paletteColors[i];
		if ((color->red != color->green) || (color->red != color->blue)) {
			return false;
		}
	}

	_grayScale = 1;
	return true;
}

bool BitmapImage::setSize(int newWidth, int newHeight) {
	if (_rawData) {
		delete [] _rawData;
	}

	_width = newWidth;
	_height = newHeight;

	if (isGrayScale()) {
		_dataSize = _width * _height;
	} else {
		_dataSize = _width * _height * sizeof(RGBA); 
	}

	_rawData = new BYTE[_dataSize]; 

	return true; 
}

void BitmapImage::generateSaltAndPepperNoise() {
	const int MAX_VALUE = 255;

	srand(time(NULL)); // time(NULL) is used as SEED for random number generator

	// Generates random number between 0 to MAX - 1
	for (int i = 0; i < _dataSize; i++) {
		double random = rand() % MAX_VALUE;

		if (random == 0 || (random == MAX_VALUE - 1)) {
			_rawData[i] = random;
		}
	}
}


