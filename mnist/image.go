package mnist

import (
	"bufio"
	"image"
	"image/color"
	"image/jpeg"
	"log"
	"os"
	"strconv"
)

func encodeImage(i int, pixels []string) {
	// Decode image
	img := image.NewGray(image.Rect(0, 0, 28, 28))
	for y := 0; y < 28; y++ {
		for x := 0; x < 28; x++ {
			pixelStr := pixels[(y*28)+x]
			pixelNum, _ := strconv.Atoi(pixelStr)
			// Must "reverse" the number
			gs := -uint8(pixelNum - 255)
			clr := color.Gray{gs}
			img.Set(x, y-1, clr)
		}
	}

	f, err := os.Create("img/" + strconv.Itoa(i) + ".jpg")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	w := bufio.NewWriter(f)
	if err := jpeg.Encode(w, img, &jpeg.Options{Quality: 100}); err != nil {
		log.Fatal(err)
	}
}
