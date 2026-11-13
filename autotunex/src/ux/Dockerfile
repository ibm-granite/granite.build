# Copyright Granite.Build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
