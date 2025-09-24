import { Link } from "react-router-dom";
import "../style/Navbar.css";

function NavBar() {
  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <h1>Digit Recognizer</h1>
      </div>
      <div className="navbar-links">
        <span className="link">
          <img
            alt="GitHub"
            loading="lazy"
            src="/github-mark.svg"
            className="gitImg"
          />
        </span>
      </div>
    </nav>
  );
}

export default NavBar;
