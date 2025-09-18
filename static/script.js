document.addEventListener('DOMContentLoaded', () => {
            // Performance optimization using requestAnimationFrame
            requestAnimationFrame(() => {
                createStarField();
                createStarburst();
                createHandsEffect();
            });
            
            // Create interactive starfield
            function createStarField() {
                const starField = document.getElementById('starField');
                const starCount = Math.max(window.innerWidth, window.innerHeight) / 2;
                
                for (let i = 0; i < starCount; i++) {
                    const star = document.createElement('div');
                    star.className = 'star';
                    
                    // Vary star sizes
                    const size = Math.random() * 2 + 1;
                    star.style.width = `${size}px`;
                    star.style.height = `${size}px`;
                    
                    // Random positioning
                    star.style.left = `${Math.random() * 100}%`;
                    star.style.top = `${Math.random() * 100}%`;
                    
                    // Vary the twinkle animation
                    star.style.animationDelay = `${Math.random() * 5}s`;
                    star.style.animationDuration = `${Math.random() * 3 + 2}s`;
                    
                    starField.appendChild(star);
                }
            }
            
            // Create ASL-inspired starburst effect
function createStarburst() {
    const starburst = document.getElementById('starburst');
    const rayCount = Math.min(100, window.innerWidth / 15);
    
    for (let i = 0; i < rayCount; i++) {
        // Create ray
        const ray = document.createElement('div');
        ray.className = 'ray';
        
        // Calculate angle and length
        const angle = (i / rayCount) * 360;
        const length = Math.random() * (window.innerWidth / 2) + (window.innerWidth / 4);
        
        ray.style.setProperty('--ray-length', `${length}px`);
        ray.style.transform = `rotate(${angle}deg)`;
        ray.style.animationDelay = `${Math.random() * 0.5}s`;
        
        starburst.appendChild(ray);
        
        // Add nodes to some rays
        if (Math.random() > 0.4) {
            const nodeCount = Math.floor(Math.random() * 2) + 1;
            
            for (let j = 0; j < nodeCount; j++) {
                const node = document.createElement('div');
                node.className = Math.random() > 0.7 ? 'node node--bright' : 'node';
                
                // Position node along ray
                const distance = Math.random() * length * 0.9 + length * 0.1;
                const x = Math.sin(angle * Math.PI / 180) * distance;
                const y = -Math.cos(angle * Math.PI / 180) * distance;
                
                node.style.left = `${x}px`;
                node.style.top = `${y}px`;
                node.style.animationDelay = `${1 + Math.random() * 1.5}s`;
                
                starburst.appendChild(node);
            }
        }
    }
}
            
            // Create hand-like points to symbolize ASL
function createHandsEffect() {
    const handContainer = document.getElementById('handContainer');
    const handCount = 20;
    
    // Create symbolic hand positions
    for (let i = 0; i < handCount; i++) {
        const hand = document.createElement('div');
        hand.className = 'hand';
        
        // Position hands primarily in the visible area
        const x = Math.random() * 80 + 10; // 10% to 90% of width
        const y = Math.random() * 80 + 10; // 10% to 90% of height
        
        hand.style.left = `${x}%`;
        hand.style.top = `${y}%`;
        hand.style.animationDelay = `${2 + Math.random() * 2}s`;
        
        handContainer.appendChild(hand);
    }
}
            
// Handle window resize for responsiveness
let resizeTimeout;
window.addEventListener('resize', () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
        const starField = document.getElementById('starField');
        const starburst = document.getElementById('starburst');
        const handContainer = document.getElementById('handContainer');
        
        starField.innerHTML = '';
        starburst.innerHTML = '';
        handContainer.innerHTML = '';
        
        createStarField();
        createStarburst();
        createHandsEffect();
    }, 250);
});
});