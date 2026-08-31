# UX-only image: builds the SvelteKit app and serves the production preview.
FROM node:20-slim

ARG AUTOTUNEX_API_URL

# Set working directory
WORKDIR /app/ux

# Install Node.js dependencies (layer cached unless package files change)
COPY package.json package-lock.json ./
RUN npm install

# Copy application files
COPY . .
RUN echo "PUBLIC_AUTOTUNEX_API_URL=$AUTOTUNEX_API_URL" >> .env
RUN npm run build

# Expose the preview server port (see vite.config.ts -> preview.port)
EXPOSE 3400

# Serve the production build
CMD ["npm", "run", "preview"]
