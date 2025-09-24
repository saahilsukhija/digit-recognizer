import { useRef, useEffect, useState } from "react";
import "../style/DrawingCanvas.css";

function DrawingCanvas({ setDigit }) {
  const canvasRef = useRef(null);

  const [isDrawing, setIsDrawing] = useState(false);
  const [isPen, setIsPen] = useState(true);

  const selectPen = () => {
    setIsPen(true);
    console.log("Pen selected");
  };

  const selectEraser = () => {
    setIsPen(false);
    console.log("Eraser selected");
  };

  const submitButtonClicked = async () => {
    if (!canvasRef.current) return;

    // Scale the canvas to 28x28 before sending

    const ctx = canvasRef.current.getContext("2d");
    const imageData = ctx.getImageData(0, 0, 28, 28); // get RGBA pixels
    let pixels = Array.from(imageData.data, (v) => v / 255);

    let grayscale = [];
    console.log(pixels)
    for (let i = 0; i < pixels.length; i += 4) {
      const avg = pixels[i + 3];
      grayscale.push(avg);
    }
    console.log(grayscale);

    try {
      const response = await fetch("http://127.0.0.1:8000/api/predict/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ pixels: grayscale }), // flatten array
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const result = await response.json();
      console.log("Digit detected:", result);
      setDigit(result.message);
    } catch (error) {
      console.error("Error detecting digit:", error);
    }
  };

  const clear = () => {
    const ctx = canvasRef.current.getContext("2d");
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    canvas.width = 28;
    canvas.height = 28;

    const ctx = canvas.getContext("2d");

    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.lineCap = "round";
    ctx.strokeStyle = isPen ? "black" : "white";
    ctx.lineWidth = 1;
  }, []); // Renders it on page load

  const startDrawing = (e) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    const scaleX = canvas.width / e.target.clientWidth;
    const scaleY = canvas.height / e.target.clientHeight;
    const x = e.nativeEvent.offsetX * scaleX;
    const y = e.nativeEvent.offsetY * scaleY;

    ctx.beginPath();
    ctx.moveTo(x, y);
    setIsDrawing(true);
  };

  const draw = (e) => {
    if (!isDrawing) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    // scale mouse coordinates from 256x256 to 28x28
    const scaleX = canvas.width / e.target.clientWidth;
    const scaleY = canvas.height / e.target.clientHeight;
    const x = e.nativeEvent.offsetX * scaleX;
    const y = e.nativeEvent.offsetY * scaleY;

    ctx.strokeStyle = isPen ? "black" : "white";
    ctx.lineWidth = isPen ? 1 : 3; // logical width for 28x28
    ctx.lineCap = "round";
    ctx.lineTo(x, y);
    ctx.stroke();
  };

  const stopDrawing = () => {
    const ctx = canvasRef.current.getContext("2d");
    ctx.closePath();
    setIsDrawing(false);
  };

  return (
    <div className="canvas-board">
      <canvas
        ref={canvasRef}
        style={{ border: "1px solid black", cursor: "crosshair" }}
        onMouseDown={startDrawing}
        onMouseMove={draw}
        onMouseUp={stopDrawing}
        onMouseLeave={stopDrawing}
      />
      <div className="canvas-toolbar">
        <div className="canvas-toolbar-tools">
          <button onClick={selectPen} className={isPen ? "selected" : ""}>
            <img src="/pen-icon.svg" />
          </button>
          <button onClick={selectEraser} className={!isPen ? "selected" : ""}>
            <img src="/eraser-icon.svg" />
          </button>
        </div>
        <button onClick={clear} className="canvas-toolbar-clear">
          Clear
        </button>
        <button onClick={submitButtonClicked} className="canvas-toolbar-clear">
          Submit
        </button>
      </div>
    </div>
  );
}

export default DrawingCanvas;
