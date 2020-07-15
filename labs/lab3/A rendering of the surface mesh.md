A rendering of the surface mesh.

We begin with the equation for the wave height $h(\vec{x},t)$ at a location, $ \vec{x} = (x,z) $, and time, $t$,

$$
\begin{align*}
h(\vec{x},t) &= \sum_\vec{k} \tilde{h}(\vec{k},t)\exp(i\vec{k}\cdot\vec{x})
\end{align*}
$$

We define $\vec{k}=(k_x,k_z)$ as,

$$
\begin{align*}
k_x &= \frac{2\pi n}{L_x}\\
k_z &= \frac{2\pi m}{L_z}\\
\end{align*}
$$


where,
$$
\begin{align*}
-\frac{N}{2} \le &n < \frac{N}{2} \\ -\frac{M}{2} \le &m < \frac{M}{2} \\ \end{align*} 
$$




In our implementation our indices, $n'$ and $m'$, will run from ![0,1,...,N-1](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_9b623fc33c5528891cdd92728e681a9a.gif) and $0,1,...,M-1$, respectively. So we have, \begin{align*} 0 \le &n' < N\\ 0 \le &m' < M\\ \end{align*} Thus, ![n](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_7b8b965ad4bca0e41ab51de7b31363a1.gif) and ![m](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_6f8f57715090da2632453988d9a1501b.gif) can be written in terms of ![n'](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_8b072dc4ba86c6008887fd8004865c61.gif) and ![m'](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_c3b1fe1f90bfbf08147e9e8fa2e5ccca.gif),  
$$
\begin{align*} n &= n'-\frac{N}{2}\\ m &= m'-\frac{M}{2}\\ \end{align*}
$$


 ![\vec{k}](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_2b67beeefbbbe81907e2fc881c4974c1.gif) in terms of ![n'](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_8b072dc4ba86c6008887fd8004865c61.gif) and ![m'](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_c3b1fe1f90bfbf08147e9e8fa2e5ccca.gif), $\begin{align*} k_x &= \frac{2\pi (n'-\frac{N}{2})}{L_x}\\ &= \frac{2\pi n'- \pi N}{L_x}\\ k_z &= \frac{2\pi (m'-\frac{M}{2})}{L_z}\\ &= \frac{2\pi m'- \pi M}{L_z}\\ \end{align*}$ We can now write our original equation for the wave height in terms of ![n'](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_8b072dc4ba86c6008887fd8004865c61.gif) and ![m'](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_c3b1fe1f90bfbf08147e9e8fa2e5ccca.gif) as, \begin{align*} h'(x,z,t) &= \sum_{m'=0}^{M-1} \sum_{n'=0}^{N-1} \tilde{h}'(n',m',t)\exp\left(\frac{ix(2\pi n' - \pi N)}{L_x} + \frac{iz(2\pi m' - \pi M)}{L_z}\right) \end{align*} We still need to look at the function, ![\tilde{h}](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_88bcd1dcc0d13c32c151dbd707a25361.gif), as a function of ![\vec{k}](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_2b67beeefbbbe81907e2fc881c4974c1.gif) and ![t](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_e358efa489f58062f10dd7316b65649e.gif) and our equivalent function, ![\tilde{h}'](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_6e0b478bb89e3920caab2d435c2295f2.gif), as a function of our indices, ![n'](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_8b072dc4ba86c6008887fd8004865c61.gif) and ![m'](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_c3b1fe1f90bfbf08147e9e8fa2e5ccca.gif), and ![t](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_e358efa489f58062f10dd7316b65649e.gif). ![\tilde{h}(\vec{k},t)](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_54d14ea3d47eb184fafb3a3f36debc19.gif) is defined as, \begin{align*} \tilde{h}(\vec{k},t) = &\tilde{h}_0(\vec{k}) \exp(i\omega(k)t) +\\ &\tilde{h}_0^*(-\vec{k})\exp(-i\omega(k)t) \\ \tilde{h}'(n',m',t) = &\tilde{h}_0'(n',m') \exp(i\omega'(n',m')t) +\\ &{\tilde{h}_0'}^*(-n',-m')\exp(-i\omega'(n',m')t) \\ \end{align*} where ![\omega(k)](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_2f4bb3aafb34cdcdfb7ac24e4dff432c.gif) is the dispersion relation given by, $\begin{align*} \omega(k) = \sqrt{gk} \\ \end{align*}$ and ![g](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_b2f5ff47436671b6e533d8dc3614845d.gif) is the local acceleration due to gravity, nominally ![9.8\frac{m}{\sec^2}](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_f625cb920066b661aa9ca4e2ba250459.gif). In terms of our indices, $\begin{align*} \omega'(n',m') = \sqrt{g\sqrt{\left(\frac{2\pi n'- \pi N}{L_x}\right)^2+\left(\frac{2\pi m'- \pi M}{L_z}\right)^2}} \\ \end{align*} $We are almost there, but we need ![\tilde{h}_0'](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_6332a6a41e42cbaa3be1db04ea51edd9.gif) and the complex conjugate, ![{\tilde{h}_0'}^*](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_659489eaf1ef838c01a3e5c04addac99.gif), in terms of our indices, \begin{align*} \tilde{h}_0(\vec{k}) &= \frac{1}{\sqrt{2}}(\xi_r+i\xi{i})\sqrt{P_h(\vec{k})} \\ \tilde{h}_0'(n',m') &= \frac{1}{\sqrt{2}}(\xi_r+i\xi{i})\sqrt{P_h'(n',m')} \\ \end{align*} where ![\xi_r](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_0308c8c3fa7e82c1c949feca725d29f0.gif) and ![\xi_i](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_9ea888f58114108ed02fea555f01fe8c.gif) are independent gaussian random variables. ![P_h(\vec{k})](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_2cb1cd8d795bd616dbaad141fbe62e82.gif) is the Phillips spectrum given by, \begin{align*} P_h(\vec{k}) &= A\frac{\exp(-1/(kL)^2)}{k^4}|\vec{k}\cdot \vec{w}|^2 \\ \end{align*} where ![\vec{w}](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_1b6e7fc2252f4d67d24020cf8067b313.gif) is the direction of the wind and ![L=V^2/g](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_9b78fb525700778334d4c056ef4c3442.gif) is the largest possible waves arising from a continuous wind of speed ![V](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_5206560a306a2e085a437fd258eb57ce.gif). In terms of the indices, \begin{align*} P_h'(n',m') &= A\frac{\exp(-1/(kL)^2)}{k^4}|\vec{k}\cdot \vec{w}|^2 \\ \vec{k} &= \left(\frac{2\pi n'- \pi N}{L_x}, \frac{2\pi m'- \pi M}{L_z}\right) \\ k &= |\vec{k}| \\ \end{align*} We should now have everything we need to render our wave height field, but we will evaluate a displacement vector for generating choppy waves and the normal vector for our lighting calculations at each vertex. The displacement vector is given by, \begin{align*} \vec{D}(\vec{x},t) &= \sum_{\vec{k}}-i\frac{\vec{k}}{k}\tilde{h}(\vec{k},t)\exp(i\vec{k}\cdot \vec{x}) \\ \end{align*} and the normal vector, ![\vec{N}](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_2ed2b5587d7508693e4d37cb6668cc62.gif), is given by, \begin{align*} \epsilon(\vec{x},t) &= \nabla h(\vec{x},t) = \sum_{\vec{k}}i\vec{k}\tilde{h}(\vec{k},t)\exp(i\vec{k}\cdot \vec{x}) \\ \vec{N}(\vec{x},t) &= (0,1,0) - \left(\epsilon_x(\vec{x},t), 0, \epsilon_z(\vec{x},t)\right) \\ &= \left(-\epsilon_x(\vec{x},t), 1, -\epsilon_z(\vec{x},t)\right) \\ \end{align*} all of which we can rewrite in terms of the indices for ease of implementation. We'll begin our implementation by looking at our vertex structure which will double to hold our values for ![\tilde{h}_0'(n',m')](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_b672f20975e2ae886e079abe5feba1ed.gif) and ![{\tilde{h}_0'}^*(-n',-m')](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_6b9c4999527526fd84e1c488c6857c2b.gif). We will apply our wave height and displacement vector to `ox`, `oy`, and `oz` to obtain the current position, `x`, `y`, and `z`.

```
struct vertex_ocean {
	GLfloat   x,   y,   z; // vertex
	GLfloat  nx,  ny,  nz; // normal
	GLfloat   a,   b,   c; // htilde0
	GLfloat  _a,  _b,  _c; // htilde0mk conjugate
	GLfloat  ox,  oy,  oz; // original position
};
```

We have another structure to return the resulting wave height, displacement vector, and normal vector at a location, ![\vec{x}](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_b07b5abd67ee0b72f4136c82c68a0c48.gif), and time, ![t](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_e358efa489f58062f10dd7316b65649e.gif),

```
struct complex_vector_normal {	// structure used with discrete fourier transform
	complex h;		// wave height
	vector2 D;		// displacement
	vector3 n;		// normal
};
```

The following is the declaration of our `cOcean` object. There are references to the fast Fourier transform which we will cover in the [next post](https://web.archive.org/web/20160123223847/http://www.keithlantz.net/2011/11/ocean-simulation-part-two-using-the-fast-fourier-transform/). In this implementation we are assuming ![M=N](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_87bf7d5575f4af42adbb9a0d855602be.gif) and ![L=L_x=L_z](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_3c6b3bba6253c61a72a212f3cc27b946.gif) from the equations above. We declare the property, `Nplus1`, for tiling purposes in order for our vertices to connect to the tile at an adjacent location.

```
class cOcean {
  private:
	bool geometry;							// flag to render geometry or surface

	float g;								// gravity constant
	int N, Nplus1;							// dimension -- N should be a power of 2
	float A;								// phillips spectrum parameter -- affects heights of waves
	vector2 w;								// wind parameter
	float length;							// length parameter
	complex *h_tilde,						// for fast fourier transform
		*h_tilde_slopex, *h_tilde_slopez,
		*h_tilde_dx, *h_tilde_dz;
	cFFT *fft;								// fast fourier transform

	vertex_ocean *vertices;					// vertices for vertex buffer object
	unsigned int *indices;					// indicies for vertex buffer object
	unsigned int indices_count;				// number of indices to render
	GLuint vbo_vertices, vbo_indices;		// vertex buffer objects

	GLuint glProgram, glShaderV, glShaderF;	// shaders
	GLint vertex, normal, texture, light_position, projection, view, model;	// attributes and uniforms

  protected:
  public:
	cOcean(const int N, const float A, const vector2 w, const float length, bool geometry);
	~cOcean();
	void release();

	float dispersion(int n_prime, int m_prime);		// deep water
	float phillips(int n_prime, int m_prime);		// phillips spectrum
	complex hTilde_0(int n_prime, int m_prime);
	complex hTilde(float t, int n_prime, int m_prime);
	complex_vector_normal h_D_and_n(vector2 x, float t);
	void evaluateWaves(float t);
	void evaluateWavesFFT(float t);
	void render(float t, glm::vec3 light_pos, glm::mat4 Projection, glm::mat4 View, glm::mat4 Model, bool use_fft);
};
```

In our constructor we set up some resources, define our vertices and indices for the [vertex buffer objects](https://web.archive.org/web/20160123223847/http://www.opengl.org/wiki/Vertex_Buffer_Object), create our shader program, and, lastly, generate our vertex buffer objects. The destructor frees our resources, and the `release` method frees our vertex buffer objects and shader program before our [OpenGL context](https://web.archive.org/web/20160123223847/http://www.opengl.org/wiki/Creating_an_OpenGL_Context) is destroyed.

```
cOcean::cOcean(const int N, const float A, const vector2 w, const float length, const bool geometry) :
	g(9.81), geometry(geometry), N(N), Nplus1(N+1), A(A), w(w), length(length),
	vertices(0), indices(0), h_tilde(0), h_tilde_slopex(0), h_tilde_slopez(0), h_tilde_dx(0), h_tilde_dz(0), fft(0)
{
	h_tilde        = new complex[N*N];
	h_tilde_slopex = new complex[N*N];
	h_tilde_slopez = new complex[N*N];
	h_tilde_dx     = new complex[N*N];
	h_tilde_dz     = new complex[N*N];
	fft            = new cFFT(N);
	vertices       = new vertex_ocean[Nplus1*Nplus1];
	indices        = new unsigned int[Nplus1*Nplus1*10];

	int index;

	complex htilde0, htilde0mk_conj;
	for (int m_prime = 0; m_prime < Nplus1; m_prime++) {
		for (int n_prime = 0; n_prime < Nplus1; n_prime++) {
			index = m_prime * Nplus1 + n_prime;

			htilde0        = hTilde_0( n_prime,  m_prime);
			htilde0mk_conj = hTilde_0(-n_prime, -m_prime).conj();

			vertices[index].a  = htilde0.a;
			vertices[index].b  = htilde0.b;
			vertices[index]._a = htilde0mk_conj.a;
			vertices[index]._b = htilde0mk_conj.b;

			vertices[index].ox = vertices[index].x =  (n_prime - N / 2.0f) * length / N;
			vertices[index].oy = vertices[index].y =  0.0f;
			vertices[index].oz = vertices[index].z =  (m_prime - N / 2.0f) * length / N;

			vertices[index].nx = 0.0f;
			vertices[index].ny = 1.0f;
			vertices[index].nz = 0.0f;
		}
	}

	indices_count = 0;
	for (int m_prime = 0; m_prime < N; m_prime++) {
		for (int n_prime = 0; n_prime < N; n_prime++) {
			index = m_prime * Nplus1 + n_prime;

			if (geometry) {
				indices[indices_count++] = index;				// lines
				indices[indices_count++] = index + 1;
				indices[indices_count++] = index;
				indices[indices_count++] = index + Nplus1;
				indices[indices_count++] = index;
				indices[indices_count++] = index + Nplus1 + 1;
				if (n_prime == N - 1) {
					indices[indices_count++] = index + 1;
					indices[indices_count++] = index + Nplus1 + 1;
				}
				if (m_prime == N - 1) {
					indices[indices_count++] = index + Nplus1;
					indices[indices_count++] = index + Nplus1 + 1;
				}
			} else {
				indices[indices_count++] = index;				// two triangles
				indices[indices_count++] = index + Nplus1;
				indices[indices_count++] = index + Nplus1 + 1;
				indices[indices_count++] = index;
				indices[indices_count++] = index + Nplus1 + 1;
				indices[indices_count++] = index + 1;
			}
		}
	}

	createProgram(glProgram, glShaderV, glShaderF, "src/oceanv.sh", "src/oceanf.sh");
	vertex         = glGetAttribLocation(glProgram, "vertex");
	normal         = glGetAttribLocation(glProgram, "normal");
	texture        = glGetAttribLocation(glProgram, "texture");
	light_position = glGetUniformLocation(glProgram, "light_position");
	projection     = glGetUniformLocation(glProgram, "Projection");
	view           = glGetUniformLocation(glProgram, "View");
	model          = glGetUniformLocation(glProgram, "Model");

	glGenBuffers(1, &vbo_vertices);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_ocean)*(Nplus1)*(Nplus1), vertices, GL_DYNAMIC_DRAW);

	glGenBuffers(1, &vbo_indices);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_indices);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_count*sizeof(unsigned int), indices, GL_STATIC_DRAW);
}

cOcean::~cOcean() {
	if (h_tilde)		delete [] h_tilde;
	if (h_tilde_slopex)	delete [] h_tilde_slopex;
	if (h_tilde_slopez)	delete [] h_tilde_slopez;
	if (h_tilde_dx)		delete [] h_tilde_dx;
	if (h_tilde_dz)		delete [] h_tilde_dz;
	if (fft)		delete fft;
	if (vertices)		delete [] vertices;
	if (indices)		delete [] indices;
}

void cOcean::release() {
	glDeleteBuffers(1, &vbo_indices);
	glDeleteBuffers(1, &vbo_vertices);
	releaseProgram(glProgram, glShaderV, glShaderF);
}
```

Below are our helper methods for evaluating the disperson relation and the Phillips spectrum in addition to ![\tilde{h}_0'](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_6332a6a41e42cbaa3be1db04ea51edd9.gif) and ![\tilde{h}'](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_6e0b478bb89e3920caab2d435c2295f2.gif).

```
float cOcean::dispersion(int n_prime, int m_prime) {
	float w_0 = 2.0f * M_PI / 200.0f;
	float kx = M_PI * (2 * n_prime - N) / length;
	float kz = M_PI * (2 * m_prime - N) / length;
	return floor(sqrt(g * sqrt(kx * kx + kz * kz)) / w_0) * w_0;
}

float cOcean::phillips(int n_prime, int m_prime) {
	vector2 k(M_PI * (2 * n_prime - N) / length,
		  M_PI * (2 * m_prime - N) / length);
	float k_length  = k.length();
	if (k_length < 0.000001) return 0.0;

	float k_length2 = k_length  * k_length;
	float k_length4 = k_length2 * k_length2;

	float k_dot_w   = k.unit() * w.unit();
	float k_dot_w2  = k_dot_w * k_dot_w;

	float w_length  = w.length();
	float L         = w_length * w_length / g;
	float L2        = L * L;
	
	float damping   = 0.001;
	float l2        = L2 * damping * damping;

	return A * exp(-1.0f / (k_length2 * L2)) / k_length4 * k_dot_w2 * exp(-k_length2 * l2);
}

complex cOcean::hTilde_0(int n_prime, int m_prime) {
	complex r = gaussianRandomVariable();
	return r * sqrt(phillips(n_prime, m_prime) / 2.0f);
}

complex cOcean::hTilde(float t, int n_prime, int m_prime) {
	int index = m_prime * Nplus1 + n_prime;

	complex htilde0(vertices[index].a,  vertices[index].b);
	complex htilde0mkconj(vertices[index]._a, vertices[index]._b);

	float omegat = dispersion(n_prime, m_prime) * t;

	float cos_ = cos(omegat);
	float sin_ = sin(omegat);

	complex c0(cos_,  sin_);
	complex c1(cos_, -sin_);

	complex res = htilde0 * c0 + htilde0mkconj * c1;

	return htilde0 * c0 + htilde0mkconj*c1;
}
```

We have the method, `h_D_and_n`, to evaluate the wave height, displacement vector, and normal vector at a location, ![\vec{x}](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_b07b5abd67ee0b72f4136c82c68a0c48.gif).

```
complex_vector_normal cOcean::h_D_and_n(vector2 x, float t) {
	complex h(0.0f, 0.0f);
	vector2 D(0.0f, 0.0f);
	vector3 n(0.0f, 0.0f, 0.0f);

	complex c, res, htilde_c;
	vector2 k;
	float kx, kz, k_length, k_dot_x;

	for (int m_prime = 0; m_prime < N; m_prime++) {
		kz = 2.0f * M_PI * (m_prime - N / 2.0f) / length;
		for (int n_prime = 0; n_prime < N; n_prime++) {
			kx = 2.0f * M_PI * (n_prime - N / 2.0f) / length;
			k = vector2(kx, kz);

			k_length = k.length();
			k_dot_x = k * x;

			c = complex(cos(k_dot_x), sin(k_dot_x));
			htilde_c = hTilde(t, n_prime, m_prime) * c;

			h = h + htilde_c;

			n = n + vector3(-kx * htilde_c.b, 0.0f, -kz * htilde_c.b);

			if (k_length < 0.000001) continue;
			D = D + vector2(kx / k_length * htilde_c.b, kz / k_length * htilde_c.b);
		}
	}
	
	n = (vector3(0.0f, 1.0f, 0.0f) - n).unit();

	complex_vector_normal cvn;
	cvn.h = h;
	cvn.D = D;
	cvn.n = n;
	return cvn;
}
```

Below we have a method to evaluate our waves using the `h_D_and_n` method. Note the tiling under the conditions, ![n'=m'=0](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_5e296cff654895c3d558fdb742352270.gif), ![n'=0](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_472418d747956bb4058abc77792cd385.gif), and ![m'=0](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_784c6f6b53daa151d5440822228c5f45.gif).

```
void cOcean::evaluateWaves(float t) {
	float lambda = -1.0;
	int index;
	vector2 x;
	vector2 d;
	complex_vector_normal h_d_and_n;
	for (int m_prime = 0; m_prime < N; m_prime++) {
		for (int n_prime = 0; n_prime < N; n_prime++) {
			index = m_prime * Nplus1 + n_prime;

			x = vector2(vertices[index].x, vertices[index].z);

			h_d_and_n = h_D_and_n(x, t);

			vertices[index].y = h_d_and_n.h.a;

			vertices[index].x = vertices[index].ox + lambda*h_d_and_n.D.x;
			vertices[index].z = vertices[index].oz + lambda*h_d_and_n.D.y;

			vertices[index].nx = h_d_and_n.n.x;
			vertices[index].ny = h_d_and_n.n.y;
			vertices[index].nz = h_d_and_n.n.z;

			if (n_prime == 0 && m_prime == 0) {
				vertices[index + N + Nplus1 * N].y = h_d_and_n.h.a;
			
				vertices[index + N + Nplus1 * N].x = vertices[index + N + Nplus1 * N].ox + lambda*h_d_and_n.D.x;
				vertices[index + N + Nplus1 * N].z = vertices[index + N + Nplus1 * N].oz + lambda*h_d_and_n.D.y;

				vertices[index + N + Nplus1 * N].nx = h_d_and_n.n.x;
				vertices[index + N + Nplus1 * N].ny = h_d_and_n.n.y;
				vertices[index + N + Nplus1 * N].nz = h_d_and_n.n.z;
			}
			if (n_prime == 0) {
				vertices[index + N].y = h_d_and_n.h.a;
			
				vertices[index + N].x = vertices[index + N].ox + lambda*h_d_and_n.D.x;
				vertices[index + N].z = vertices[index + N].oz + lambda*h_d_and_n.D.y;

				vertices[index + N].nx = h_d_and_n.n.x;
				vertices[index + N].ny = h_d_and_n.n.y;
				vertices[index + N].nz = h_d_and_n.n.z;
			}
			if (m_prime == 0) {
				vertices[index + Nplus1 * N].y = h_d_and_n.h.a;
			
				vertices[index + Nplus1 * N].x = vertices[index + Nplus1 * N].ox + lambda*h_d_and_n.D.x;
				vertices[index + Nplus1 * N].z = vertices[index + Nplus1 * N].oz + lambda*h_d_and_n.D.y;
				
				vertices[index + Nplus1 * N].nx = h_d_and_n.n.x;
				vertices[index + Nplus1 * N].ny = h_d_and_n.n.y;
				vertices[index + Nplus1 * N].nz = h_d_and_n.n.z;
			}
		}
	}
}
```

Finally, we have our `render` method. We have a static variable, `eval`. Under the condition that we are not using the fast Fourier transform, we simply update our vertices and normals during the first frame. This will permit us to navigate the first frame of our wave height field in our `main` function. We then specify the uniforms and attribute offsets for our shader program and then render either the surface or the geometry.

```
void cOcean::render(float t, glm::vec3 light_pos, glm::mat4 Projection, glm::mat4 View, glm::mat4 Model, bool use_fft) {
	static bool eval = false;
	if (!use_fft && !eval) {
		eval = true;
		evaluateWaves(t);
	} else if (use_fft) {
		evaluateWavesFFT(t);
	}

	glUseProgram(glProgram);
	glUniform3f(light_position, light_pos.x, light_pos.y, light_pos.z);
	glUniformMatrix4fv(projection, 1, GL_FALSE, glm::value_ptr(Projection));
	glUniformMatrix4fv(view,       1, GL_FALSE, glm::value_ptr(View));
	glUniformMatrix4fv(model,      1, GL_FALSE, glm::value_ptr(Model));

	glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices);
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertex_ocean) * Nplus1 * Nplus1, vertices);
	glEnableVertexAttribArray(vertex);
	glVertexAttribPointer(vertex, 3, GL_FLOAT, GL_FALSE, sizeof(vertex_ocean), 0);
	glEnableVertexAttribArray(normal);
	glVertexAttribPointer(normal, 3, GL_FLOAT, GL_FALSE, sizeof(vertex_ocean), (char *)NULL + 12);
	glEnableVertexAttribArray(texture);
	glVertexAttribPointer(texture, 3, GL_FLOAT, GL_FALSE, sizeof(vertex_ocean), (char *)NULL + 24);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_indices);
	for (int j = 0; j < 10; j++) {
		for (int i = 0; i < 10; i++) {
			Model = glm::scale(glm::mat4(1.0f), glm::vec3(5.f, 5.f, 5.f));
			Model = glm::translate(Model, glm::vec3(length * i, 0, length * -j));
			glUniformMatrix4fv(model, 1, GL_FALSE, glm::value_ptr(Model));
			glDrawElements(geometry ? GL_LINES : GL_TRIANGLES, indices_count, GL_UNSIGNED_INT, 0);
		}
	}
}
```

In the code above there are references to the `complex`, `vector2`, and `vector3` objects. These are available in the download. Any library that supports complex variables in addition to 2-dimensional and 3-dimensional vectors would be appropriate. Additionally, we have used [Euler's formula](https://web.archive.org/web/20160123223847/http://en.wikipedia.org/wiki/Euler's_formula) in our code above. Euler's formula states for any real ![x](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_9dd4e461268c8034f5c8564e155c67a6.gif),

\begin{align*}
\exp(ix) &= \cos x + i\sin x \\
\end{align*}

This post is simply a brute force evaluation of our ocean surface using the discrete Fourier transform. In the [following post](https://web.archive.org/web/20160123223847/http://www.keithlantz.net/2011/11/ocean-simulation-part-two-using-the-fast-fourier-transform/) we will provide a deeper analysis of the equations used here to implement our own fast Fourier transform algorithm. The download for this project includes a fast Fourier transform, so have a look at it and we'll analyze it in the next post.

I've added a mouse handler in a vain similar to the [keyboard handler](https://web.archive.org/web/20160123223847/http://www.keithlantz.net/2011/10/a-keyboard-handler-and-timer-in-c-for-the-linux-platform/) and [joystick handler](https://web.archive.org/web/20160123223847/http://www.keithlantz.net/2011/10/a-linux-c-joystick-object/) from previous posts. The source code expects the keyboard device node to be located at `/dev/input/event0` and the device node for the mouse to be located at `/dev/input/event1`. You can navigate using the mouse and the WASD keys.

One last thing to mention is in regards to the fog factor. Below is the [vertex shader](https://web.archive.org/web/20160123223847/http://en.wikipedia.org/wiki/Vertex_shader) which is reminiscent of our previous shaders with the addition of the `fog_factor` variable. We simply divide the z coordinate of our vertex position in view space to retrieve a fog factor in the interval ![[0,1]](https://web.archive.org/web/20160123223847im_/http://www.keithlantz.net/wp-content/plugins/latex/cache/tex_ccfcd347d0bf65dc77afe01a3306a96b.gif).

```
#version 330

in vec3 vertex;
in vec3 normal;
in vec3 texture;

uniform mat4 Projection;
uniform mat4 View;
uniform mat4 Model;
uniform vec3 light_position;

out vec3 light_vector;
out vec3 normal_vector;
out vec3 halfway_vector;
out float fog_factor;
out vec2 tex_coord;

void main() {
	gl_Position = View * Model * vec4(vertex, 1.0);
	fog_factor = min(-gl_Position.z/500.0, 1.0);
	gl_Position = Projection * gl_Position;

	vec4 v = View * Model * vec4(vertex, 1.0);
	vec3 normal1 = normalize(normal);

	light_vector = normalize((View * vec4(light_position, 1.0)).xyz - v.xyz);
	normal_vector = (inverse(transpose(View * Model)) * vec4(normal1, 0.0)).xyz;
        halfway_vector = light_vector + normalize(-v.xyz);

	tex_coord = texture.xy;
}
```

In the [fragment shader](https://web.archive.org/web/20160123223847/http://en.wikipedia.org/wiki/Pixel_shader) we simply apply the fog factor to the final fragment color.

```
#version 330

in vec3 normal_vector;
in vec3 light_vector;
in vec3 halfway_vector;
in vec2 tex_coord;
in float fog_factor;
uniform sampler2D water;
out vec4 fragColor;

void main (void) {
	//fragColor = vec4(1.0, 1.0, 1.0, 1.0);

	vec3 normal1         = normalize(normal_vector);
	vec3 light_vector1   = normalize(light_vector);
	vec3 halfway_vector1 = normalize(halfway_vector);

	vec4 c = vec4(1,1,1,1);//texture(water, tex_coord);

	vec4 emissive_color = vec4(1.0, 1.0, 1.0,  1.0);
	vec4 ambient_color  = vec4(0.0, 0.65, 0.75, 1.0);
	vec4 diffuse_color  = vec4(0.5, 0.65, 0.75, 1.0);
	vec4 specular_color = vec4(1.0, 0.25, 0.0,  1.0);

	float emissive_contribution = 0.00;
	float ambient_contribution  = 0.30;
	float diffuse_contribution  = 0.30;
	float specular_contribution = 1.80;

	float d = dot(normal1, light_vector1);
	bool facing = d > 0.0;

	fragColor = emissive_color * emissive_contribution +
		    ambient_color  * ambient_contribution  * c +
		    diffuse_color  * diffuse_contribution  * c * max(d, 0) +
                    (facing ?
			specular_color * specular_contribution * c * max(pow(dot(normal1, halfway_vector1), 120.0), 0.0) :
			vec4(0.0, 0.0, 0.0, 0.0));

	fragColor = fragColor * (1.0-fog_factor) + vec4(0.25, 0.75, 0.65, 1.0) * (fog_factor);

	fragColor.a = 1.0;
}
```

If you have any questions or comments, let me know. We will continue with this project in the [next post](https://web.archive.org/web/20160123223847/http://www.keithlantz.net/2011/11/ocean-simulation-part-two-using-the-fast-fourier-transform/).

Download this project: [waves.dft_correction20121002.tar.bz2](https://web.archive.org/web/20160123223847/http://www.keithlantz.net/wp-content/uploads/2012/10/waves.dft_correction20121002.tar.bz2)

References:
\1. Tessendorf, Jerry. Simulating Ocean Water. *In SIGGRAPH 2002 Course Notes #9 (Simulating Nature: Realistic and Interactive Techniques)*, ACM Press.