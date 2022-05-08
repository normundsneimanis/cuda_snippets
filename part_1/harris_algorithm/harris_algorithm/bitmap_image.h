/*
 * Harris corners detector algorithm on GPU
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
	RGBA(BYTE r, BYTE g, BYTE b, BYTE a = 0) : blue(b), green(g), red(r), alpha(a) {}

	BYTE getGrayScale() const { return (red + green + blue) / 3; }
	void setGrayScale(BYTE v) { blue = v; green = v; red = v; alpha = 0; }
	//void set(const RGBA& color) { blue = color.blue; green = color.green; red = color.red; alpha = color. }
	
	static const RGBA Red;
	static const RGBA Green;
	static const RGBA Blue;
	static const RGBA Black;
}; 

bool operator != (const RGBA& l, const RGBA& r);
bool operator < (const RGBA& l, const RGBA& r);

typedef std::map<RGBA, BYTE> ColorTable;

class BitmapImage {
public:
	BitmapImage() : _paletteColors(NULL), _rawData(NULL), _grayScale(-1) {}
	BitmapImage(const BitmapImage& input);
	~BitmapImage();
	
	bool loadFromFile(const std::string& fileName);
	bool saveToFile(const std::string& fileName);
	
	int getNumberOfColors(void) const;
	BYTE* getRawData() const { return _rawData; } 
	int getDataSize() const { return _dataSize; }
	int getWidth() const { return _width; }
	int getHeight() const { return _height; }
	int getStride() const { return _stride; }
	RGBA* getPaletteColors() const { return _paletteColors; }
	void copy32bpp(const BitmapImage& source);
	
	void generateSaltAndPepperNoise();

private:
	void setBitDepth(int newDepth); 
	bool createStandardColorTable(void);
	bool setSize(int newWidth, int newHeight);
	bool isGrayScale() const;
	void calculateStride();

	bool read8bitRow(BYTE* buffer, int bufferSize, int row, bool grayScale);
	bool read24bitRow(BYTE* Buffer, int bufferSize, int row);
	
	BYTE findClosestColor(const RGBA& input);
	bool write8bitRow(BYTE* buffer, int bufferSize, int row);
	bool write24bitRow(BYTE* buffer, int bufferSize, int row);
	bool write32bitRow(BYTE* buffer, int bufferSize, int row);
	
	int _dataSize;
	int _bitDepth;
	RGBA* _paletteColors;
	int _width;
	int _height;
	
	// pointer to image data 
	BYTE* _rawData;
		
	// image stride (line size)
	int _stride;
	
	mutable int _grayScale;
	ColorTable _colorTable;
};