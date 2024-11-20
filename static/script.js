document.getElementById('recommendation-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const formData = {
        state: document.getElementById('state').value,
        type: document.getElementById('type').value,
        best_time: document.getElementById('best_time').value,
        popularity: document.getElementById('popularity').value
    };

    const response = await fetch('/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
    });

    const resultDiv = document.getElementById('recommendation-result');
    resultDiv.innerHTML = ""; // Clear previous results

    if (response.ok) {
        const result = await response.json();
        resultDiv.innerHTML = `
            <h2>Recommended Destination: <br> ${result.destination}</h2>
            
        `;
    } else {
        const error = await response.json();
        resultDiv.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
    }
});
