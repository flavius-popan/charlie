/**
 * Charlie Navigation JavaScript
 *
 * Basic client-side navigation enhancements.
 * Will be extended with AJAX and interactive features.
 */

(function () {
  "use strict";

  // Log page load
  console.log("Charlie navigation loaded");

  // Smooth scroll to top
  function scrollToTop() {
    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });
  }

  // Add active state to navigation links
  function setActiveNav() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll(".site-nav a");

    navLinks.forEach((link) => {
      if (
        link.getAttribute("href") === currentPath ||
        (currentPath === "/" && link.getAttribute("href") === "/")
      ) {
        link.style.backgroundColor = "rgba(255, 255, 255, 0.2)";
      }
    });
  }

  // Initialize on DOM ready
  document.addEventListener("DOMContentLoaded", function () {
    setActiveNav();

    // Add keyboard shortcuts
    document.addEventListener("keydown", function (e) {
      // Press 'h' to go home
      if (e.key === "h" && !e.ctrlKey && !e.metaKey && !e.altKey) {
        const activeElement = document.activeElement;
        if (
          activeElement.tagName !== "INPUT" &&
          activeElement.tagName !== "TEXTAREA"
        ) {
          window.location.href = "/";
        }
      }
    });

    // Log successful initialization
    console.log("Charlie navigation initialized");
  });
})();
