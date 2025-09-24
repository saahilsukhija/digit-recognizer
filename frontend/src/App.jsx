import "./style/App.css";
import NavBar from "./components/NavBar.jsx";
import DrawingCanvas from "./components/DrawingCanvas";
import Output from "./components/Output";
import { useState } from "react";

function App() {
  const [digit, setDigit] = useState(null);

  return (
    <>
      <NavBar />
      <main className="main-content">
        <DrawingCanvas setDigit={setDigit}/>
        <Output number={digit}/>
      </main>
    </>
  );
}

export default App;
