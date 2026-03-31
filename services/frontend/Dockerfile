FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --prefer-offline || npm install

COPY . .

ENV NODE_ENV=development
ENV NEXT_PUBLIC_API_URL=http://localhost:8000

EXPOSE 3000

CMD ["npm", "run", "dev"]
