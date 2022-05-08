/*
 * Salt-and-pepper noise filtering on GPU
 * Arnis Lektauers 2019 Riga Technical University
 */
#pragma once

#include <string>
#include <map>

typedef unsigned char BYTE;
typedef unsigned short WORD;
typedef unsigned int DWORD;

#define DEF_X_PELS_PER_METER 3780
#define DEF_Y_PELS_PER_METER 3780

#define MAX_PALETTE_SIZE 256

struct RGBA {
	BYTE blue;
	BYTE green;
	BYTE red;
	BYTE alpha;

	RGBA() : alpha(0) {}
	RGBA(BYTE r, BYTE g, BYTE b, BYTE a) : blue(b), green(g), red(r), alpha(a) {}
	RGBA(BYTE v) : blue(v), green(v), red(v), alpha(0) {}

	BYTE getGrayScale() const { return (red + green + blue) / 3; }
}; 

bool operator < (const RGBA& l, const RGBA& r);

typedef std::map<RGBA, BYTE> ColorTable;

class BitmapImage {
public:
	BitmapImage() : _paletteColors(NULL), _rawData(NULL), _grayScale(-1) {}
	BitmapImage(const BitmapImage& input, bool copyData);
	~BitmapImage();
	
	bool loadFromFile(const std::string& fileName);
	bool saveToFile(const std::string& fileName);
	
	int getNumberOfColors(void) const;
	BYTE* getRawData() const { return _rawData; } 
	size_t getDataSize() const { return _dataSize; }
	int getWidth() const { return _width; }
	int getHeight() const { return _height; }
	RGBA* getPaletteColors() const { return _paletteColors; }

	void generateSaltAndPepperNoise();

private:
	void setBitDepth(int newDepth); 
	bool createStandardColorTable(void);
	bool setSize(int newWidth , int newHeight);
	bool isGrayScale() const;

	bool read8bitRow(BYTE* buffer, int bufferSize, int row, bool grayScale);
	bool read24bitRow(BYTE* Buffer, int bufferSize, int row);
	bool read32bitRow(BYTE* buffer, int bufferSize, int row);

	BYTE findClosestColor(const RGBA& input);
	bool write8bitRow(BYTE* buffer, int bufferSize, int row);
	bool write24bitRow(BYTE* buffer, int bufferSize, int row);
	bool write32bitRow(BYTE* buffer, int bufferSize, int row);

	size_t _dataSize;
	int _bitDepth;
	BYTE* _rawData;
	RGBA* _paletteColors;
	int _width;
	int _height;
	mutable int _grayScale;
	ColorTable _colorTable;
};