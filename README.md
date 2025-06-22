# Getting Started with Create React App

This project was bootstrapped with [Create React App](https://github.com/facebook/create-react-app).

## Available Scripts

In the project directory, you can run:

### `npm start`

Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in your browser.

The page will reload when you make changes.\
You may also see any lint errors in the console.

### `npm test`

Launches the test runner in the interactive watch mode.\
See the section about [running tests](https://facebook.github.io/create-react-app/docs/running-tests) for more information.

### `npm run build`

Builds the app for production to the `build` folder.\
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.\
Your app is ready to be deployed!

See the section about [deployment](https://facebook.github.io/create-react-app/docs/deployment) for more information.

### `npm run eject`

**Note: this is a one-way operation. Once you `eject`, you can't go back!**

If you aren't satisfied with the build tool and configuration choices, you can `eject` at any time. This command will remove the single build dependency from your project.

Instead, it will copy all the configuration files and the transitive dependencies (webpack, Babel, ESLint, etc) right into your project so you have full control over them. All of the commands except `eject` will still work, but they will point to the copied scripts so you can tweak them. At this point you're on your own.

You don't have to ever use `eject`. The curated feature set is suitable for small and middle deployments, and you shouldn't feel obligated to use this feature. However we understand that this tool wouldn't be useful if you couldn't customize it when you are ready for it.

## Learn More

You can learn more in the [Create React App documentation](https://facebook.github.io/create-react-app/docs/getting-started).

To learn React, check out the [React documentation](https://reactjs.org/).

### Code Splitting

This section has moved here: [https://facebook.github.io/create-react-app/docs/code-splitting](https://facebook.github.io/create-react-app/docs/code-splitting)

### Analyzing the Bundle Size

This section has moved here: [https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size](https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size)

### Making a Progressive Web App

This section has moved here: [https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app](https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app)

### Advanced Configuration

This section has moved here: [https://facebook.github.io/create-react-app/docs/advanced-configuration](https://facebook.github.io/create-react-app/docs/advanced-configuration)

### Deployment

This section has moved here: [https://facebook.github.io/create-react-app/docs/deployment](https://facebook.github.io/create-react-app/docs/deployment)

### `npm run build` fails to minify

This section has moved here: [https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify](https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify)

🎧 FinOps at Spotify: From Cloud Cost Spikes to Engineering Ownership
Spotify is a prime example of how a fast-scaling company can integrate FinOps practices deeply into its engineering culture—transforming rising cloud costs into measurable business efficiency. Here's a breakdown of their approach and outcomes.

🔍 The Problem
Scaling on the cloud with fragmented ownership
Spotify relies entirely on Google Cloud Platform (GCP) to support over 299 million users globally. With its squad-based engineering structure, each team operates independently—leading to decentralized cloud spend and limited financial oversight.

Lack of real-time visibility
Traditional cloud billing didn’t provide squad-level insights. Teams couldn’t easily answer “who is spending what” or “what are we getting in return,” making optimization difficult.

Escalating costs without accountability
As growth accelerated, engineering teams spun up resources quickly. Without guardrails or awareness, cloud costs increased rapidly without direct responsibility tied to those driving the spend.

“The default state of cloud costs is chaos unless made visible and actionable.” – Spotify Engineering

🛠 FinOps Practices Applied
🎛 1. Cost Insights Plugin for Backstage
Spotify built a custom plugin called Cost Insights, embedded into their internal developer portal (Backstage).

It visualizes cloud spend per team, product, and service, directly inside the tools developers use daily.

Teams can view trends like cost per daily active user (DAU), helping relate spending to product growth.

Engineers use this insight to optimize usage patterns—turn off idle resources, switch storage classes, or fine-tune autoscaling.

“Engineers are natural optimizers—just give them the right data.”

🚨 2. Automated Cost Alerting
Spotify set threshold-based alerts within Cost Insights.

When a team’s cloud bill exceeds expected growth, they receive proactive alerts to review usage.

The Cost Engineering team only steps in if a squad’s cost-per-user metric deviates heavily from growth.

“Our goal wasn’t to punish spend—but to ensure we spend smarter.”

📚 3. “Our Cookbook”: Internal FinOps Knowledge Sharing
Spotify developed a collaborative internal wiki, known as Our Cookbook, where engineers document successful cost optimizations.

This includes real stories about right-sizing, optimizing Kubernetes clusters, and storage class changes.

It fosters a culture of peer learning, turning cost-saving into a friendly engineering challenge.

“Engineers began to share wins like badges of honor—FinOps became fun.”

📊 4. Unit Economics and KPI-Driven Accountability
Spotify introduced unit economics metrics like:

Cost per search

Cost per daily active user (DAU)

These KPIs tie cloud spend to business impact. Cost Insights enables teams to view whether their cost-per-unit is increasing or decreasing, and react accordingly.

Teams with rising costs but flat usage are prioritized for review. Growing teams with proportionate cost growth are left alone.

“This changed the conversation from ‘you’re over budget’ to ‘what value are you delivering?’”

✅ Impact and Results
Cloud savings of 15–30% annually
Spotify reported significant GCP savings by improving visibility and aligning cost with team-level action. Cost Insights played a major role in this transformation .

Savings enabled 25 new squads
Spotify reinvested cloud savings into building new teams. According to internal reporting, early FinOps gains funded the equivalent of 25 new squads across the company .

Improved engineering ownership
Engineers gained real-time cost responsibility, resulting in smarter infrastructure decisions and better system design.

Increased reliability alongside cost efficiency
Optimization efforts didn’t sacrifice performance. In fact, better cost management often led to improved reliability and faster systems, proving that efficiency = resilience.

“Cost visibility improved performance as much as it reduced spend.” – Spotify PM

📌 Summary
Spotify’s FinOps journey showcases a model where:

Engineering teams are empowered, not micromanaged.

Cost data is integrated directly into developer workflows.

Optimization is collaborative, measurable, and tied to business goals.

By decentralizing accountability and embedding financial insights where engineers work, Spotify turned cost management into a competitive engineering advantage—one that scales.

📝 Sources:
Spotify Engineering Blog on Cost Insights

RedMonk Report on FinOps at Spotify

Dev.to: Spotify’s GCP Cost Optimization Journey

Reddit AMA on Spotify Cost Engineering

Let me know if you'd like this formatted into a blog-ready Markdown file, added with diagrams or quotes!









